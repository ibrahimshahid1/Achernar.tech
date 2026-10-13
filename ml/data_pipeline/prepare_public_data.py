"""Reproducible public intake entry point; never starts model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .import_public_cloud import prepare_lenghu,prepare_almeria
from .import_public_astrom3 import prepare as prepare_astrom3
from .import_public_imaging import prepare_synthetic,prepare_hst
from .import_hst_corpus import prepare as prepare_hst_corpus
from .import_lenghu_bulk import prepare as prepare_bulk
from .validate_public_data import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('ml/data/public'))
    parser.add_argument('--stage',choices=('small','bulk','all'),default='small')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root
    if args.stage in ('small','all'):
        for name,function in [('lenghu_sky8_human',prepare_lenghu),('almeria_v2',prepare_almeria),
            ('astrom3_full42',prepare_astrom3),('stellar_sim_v1',prepare_synthetic),
            ('stellar_fits_pilot',prepare_hst),('hst_acs_corpus_v1',prepare_hst_corpus)]:
            if not (root/name/'manifest.json').exists():
                function(root)
    if args.stage in ('bulk','all'):
        if not (root/'lenghu_sky8_human/manifest.json').exists():prepare_lenghu(root)
        prepare_bulk(root,workers=args.workers)
    print(json.dumps(validate(root,hashes=True,require_complete=args.stage=='all'),indent=2))


if __name__=='__main__':main()
