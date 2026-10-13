"""Validate the prepared public inputs; report partial downloads explicitly."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from .import_public_cloud import CloudSegmentationDataset
from .import_public_astrom3 import NativeAstroM3Dataset
from .lenghu_bulk_dataset import LenghuTeacherDataset, prepare_pairs
from .public_common import sha256_file,write_json
from .split_registry import load_registered_splits


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_groups(records, id_field):
    ids=[r[id_field] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    groups={}
    for row in records:
        groups.setdefault(row['split_group'],set()).add(row['split'])
    if any(len(partitions)>1 for partitions in groups.values()):
        raise ValueError("A source group crosses splits")
    return len(groups)


def validate(root, *, hashes=False, require_complete=False):
    root=Path(root); results={}
    for name in ('lenghu_sky8_human','almeria_v2'):
        path=root/name; records=rows(path/'records.jsonl')
        groups=validate_groups(records,'sample_id')
        classes={0,1,2,255} if name.startswith('lenghu') else {0,1,255}
        pixel_counts={};empty_masks=0
        for split in ('train','validation','test'):
            dataset=CloudSegmentationDataset(path,split)
            counts=Counter()
            for sample in dataset:
                image,mask=sample['image'],sample['mask']
                if image.shape!=(3,512,512) or mask.shape!=(512,512) or not np.isfinite(image).all() or not set(np.unique(mask))<=classes:
                    raise ValueError('Invalid cloud payload')
                unique,n=np.unique(mask,return_counts=True)
                counts.update({str(int(k)):int(v) for k,v in zip(unique,n)})
                empty_masks+=int(np.all(mask==255))
            pixel_counts[split]=dict(counts)
            if split=='train' and not all(counts[str(k)]>0 for k in classes-{255}):raise ValueError('A required cloud class has no training pixels')
        results[name]={'status':'PASS','records':len(records),'groups':groups,
                       'splits':dict(Counter(r['split'] for r in records)),
                       'all_images_masks_loaded':True,'class_pixel_counts':pixel_counts,
                       'entirely_ignored_masks':empty_masks}
    path=root/'astrom3_full42'; manifest=json.loads((path/'manifest.json').read_text())
    records=rows(path/'records.jsonl');validate_groups(records,'sample_id')
    for shard in manifest['shards']:
        file=path/shard['path']
        if hashes and sha256_file(file)!=shard['sha256']:
            raise ValueError('AstroM3 shard content changed')
        with np.load(file,allow_pickle=False) as data:
            for key in ('photometry_values','spectra','metadata'):
                if not np.isfinite(data[key]).all():raise ValueError('Nonfinite AstroM3 values')
            if data['spectra'].shape[1:]!=(3,2575) or data['metadata'].shape[1:]!=(34,):
                raise ValueError('AstroM3 dimensions')
    for split in ('train','validation','test'):
        dataset=NativeAstroM3Dataset(path,split)
        for i in (0,len(dataset)//2,len(dataset)-1):
            sample=dataset[i]
            if sample['photometry'].shape!=(200,9) or sample['photometry_mask'].shape!=(200,):
                raise ValueError('AstroM3 adapter dimensions')
    results['astrom3_full42']={'status':'PASS','records':len(records),
        'splits':manifest['split_counts'],'all_shards_loaded':True,
        'identity_verified':False,'image_pairing_eligible':False}
    for name in ('stellar_sim_v1','hst_acs_corpus_v1'):
        path=root/name
        if not (path/'manifest.json').exists():
            results[name]={'status':'INCOMPLETE'};continue
        with np.load(path/'fqe.npz',allow_pickle=False) as data:
            splits=load_registered_splits(str(path/'fqe.splits.json'),str(path/'fqe.npz'),data['split_group'])
            if data['crops'].shape[1:]!=(64,64) or not np.isfinite(data['crops']).all() or not np.isfinite(data['log_hfr']).all():
                raise ValueError('Invalid FQE payload')
            if data['ranking_eligible'].any() or not np.allclose(np.exp(data['log_hfr']),data['hfr_px']):
                raise ValueError('Invalid FQE label contract')
            results[name]={'status':'PASS','fqe_crops':len(data['crops']),
                           'splits':{k:len(v) for k,v in splits.items()},'ranking_eligible':False}
    path=root/'stellar_sim_v1'
    with np.load(path/'ted.npz',allow_pickle=False) as data:
        if data['crops'].shape[1:]!=(32,32) or not np.isfinite(data['crops']).all():raise ValueError('Invalid TED crops')
        if not np.isfinite(data['reference_ellipticity']).all():raise ValueError('Invalid TED shape labels')
        if not np.isfinite(data['reference_pa_image_rad'][data['reference_angle_valid']]).all():raise ValueError('Invalid defined TED angles')
        results['stellar_sim_v1']['ted_observations']=len(data['crops'])
        results['stellar_sim_v1']['ted_eligible']=int(data['training_eligible'].sum())
    with np.load(path/'bge.npz',allow_pickle=False) as data:
        if data['images'].shape!=data['background_truth'].shape or not np.isfinite(data['background_truth']).all() or not (data['variance_truth_preclip']>0).all():raise ValueError('Invalid BGE truth')
        valid=(data['pixel_flags']&1)==0
        if not np.isfinite(data['images'][valid]).all():raise ValueError('Unmasked nonfinite BGE pixels')
        results['stellar_sim_v1']['bge_scenes']=len(data['images'])
    path=root/'lenghu_sky8_bulk'
    results['lenghu_sky8_bulk']={'status':'INCOMPLETE'}
    if (path/'download_receipts.json').exists():
        manifest=prepare_pairs(path)
        dataset=LenghuTeacherDataset(path)
        if len(dataset):
            for i in np.linspace(0,len(dataset)-1,min(32,len(dataset)),dtype=int):
                sample=dataset[int(i)]
                if sample['image'].shape!=(3,512,512) or sample['teacher_logits'].shape!=(3,64,64):
                    raise ValueError('Bulk teacher payload')
                if sample['content_mask'].shape!=(64,64) or sample['content_mask'].dtype!=bool:raise ValueError('Bulk geometry mask')
        results['lenghu_sky8_bulk']={'status':'PASS' if manifest['download_complete'] else 'INCOMPLETE',
            **manifest['counts'],'sampled_pairs':min(32,len(dataset)),
            'object_arrays_loaded':False}
    incomplete=any(v['status']=='INCOMPLETE' for v in results.values())
    report={'status':'INCOMPLETE' if incomplete else 'PASS','datasets':results,
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'hashes_rechecked':hashes,'hash_scope':'AstroM3 prepared shards when --hashes; FQE split registries always; source downloads verified at intake',
            'model_training_performed':False,
            'qualification':'input contracts only; not model or physical validation'}
    write_json(root/'validation_report.json',report)
    if require_complete and incomplete:raise RuntimeError('Required download or import still incomplete')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='ml/data/public')
    parser.add_argument('--hashes',action='store_true')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    print(json.dumps(validate(args.root,hashes=args.hashes,require_complete=args.require_complete),indent=2))


if __name__=='__main__':main()
