"""Finish indexes and validation after an already-running bulk transfer.

This is a finite local import job, not a scheduled automation. It never trains
models, touches hardware or changes volumes. Completed shards survive errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
from pathlib import Path
import time

from .import_hst_corpus import prepare as prepare_hst
from .import_lenghu_bulk import prepare as prepare_bulk
from .validate_public_data import validate
from .public_common import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('ml/data/public'))
    parser.add_argument('--wait-download',action='store_true')
    parser.add_argument('--max-hours',type=float,default=4)
    args=parser.parse_args();root=args.root;started=time.monotonic()
    root.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
           'root':str(root.resolve()),'model_training_performed':False}
    state_path=root/'intake_job.json'
    def status(phase,**details):
        state.update(status=phase,updated_at=datetime.now(timezone.utc).isoformat(),**details)
        write_json(state_path,state)
        print(json.dumps(state),flush=True)
    try:
        if args.wait_download:
            last_count=-1
            while True:
                path=root/'lenghu_sky8_bulk/progress.json'
                progress=json.loads(path.read_text()) if path.exists() else {}
                count=progress.get('completed_files',0)
                if count!=last_count:
                    status('waiting_for_existing_download',completed_files=count,
                           completed_bytes=progress.get('completed_bytes',0))
                    last_count=count
                manifest_path=root/'lenghu_sky8_bulk/manifest.json'
                if manifest_path.exists() and progress.get('status')!='downloading':break
                if time.monotonic()-started>args.max_hours*3600:raise TimeoutError('Bulk transfer did not finish within the finite job window; completed data retained')
                time.sleep(30)
        manifest_path=root/'lenghu_sky8_bulk/manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get('status')!='downloaded_indexed':
            status('resuming_incomplete_download')
            prepare_bulk(root,workers=4)
        if not (root/'hst_acs_corpus_v1/manifest.json').exists():
            raise RuntimeError('HST import still active or incomplete; rerun finalizer after it finishes')
        status('finalizing_hst_metadata_and_indexes')
        hst=json.loads((root/'hst_acs_corpus_v1/manifest.json').read_text())
        if 'strict_fqe_crops' not in hst:prepare_hst(root)
        status('validating_prepared_inputs')
        subprocess.run([sys.executable,'-m','ml.data_pipeline.validate_public_data','--root',str(root),
                        '--hashes','--require-complete'],check=True)
        report=json.loads((root/'validation_report.json').read_text())
        status('complete',validation_status=report['status'],finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__':main()
