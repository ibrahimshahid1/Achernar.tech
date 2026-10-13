import io
import hashlib
import json
import tarfile

import numpy as np
import pytest

from ml.data_pipeline.import_lenghu_bulk import index_tar
from ml.data_pipeline.lenghu_bulk_dataset import excluded_dates, numeric_logits, read_member,filename_timestamp,quarantine_unpaired
from ml.data_pipeline.public_common import write_jsonl


def test_numeric_teacher_loading_never_requires_object_metadata():
    buffer=io.BytesIO()
    np.savez_compressed(buffer, classes=np.array([0,1,2]),
        logits_grid=np.zeros((64,64,3),np.float32),
        class_names=np.array(["sky","cloud","contamination"],dtype=object))
    assert numeric_logits(buffer.getvalue()).shape == (64,64,3)
    buffer=io.BytesIO()
    np.savez(buffer,classes=[2,1,0],logits_grid=np.zeros((64,64,3),np.float32))
    with pytest.raises(ValueError):
        numeric_logits(buffer.getvalue())
    buffer=io.BytesIO()
    data=np.zeros((64,64,3),np.float32);data[0,0,0]=np.nan
    np.savez(buffer,classes=[0,1,2],logits_grid=data)
    with pytest.raises(ValueError):
        numeric_logits(buffer.getvalue())


def test_only_exact_reviewed_orphan_can_be_quarantined_with_original_retained(tmp_path):
    path = tmp_path / "source.tar"
    path.write_bytes(b"padfixture")
    key = ("image", ".jpg", "source.tar", 3, 7)
    reviewed = {key: hashlib.sha256(b"fixture").hexdigest()}
    ledger = quarantine_unpaired(tmp_path, [key], reviewed=reviewed)
    assert ledger[0]["training_eligible"] is False
    assert path.read_bytes() == b"padfixture"
    with pytest.raises(ValueError, match="Unreviewed"):
        quarantine_unpaired(tmp_path, [("logits", *key[1:])], reviewed=reviewed)
    path.write_bytes(b"padchanged")
    with pytest.raises(ValueError, match="payload changed"):
        quarantine_unpaired(tmp_path, [key], reviewed=reviewed)


def test_holdout_exclusion_guards_midnight_and_ignores_training(tmp_path):
    write_jsonl(tmp_path/'records.jsonl',[
       {'split':'test','source_timestamp':'2020-01-01T01:00:00'},
       {'split':'train','source_timestamp':'2020-02-01T01:00:00'}])
    assert excluded_dates(tmp_path) == {'2019-12-31','2020-01-01','2020-01-02'}


def test_producer_numeric_suffix_preserves_timestamp_grouping():
    assert filename_timestamp('2019-09-26-18-03-47_1')==filename_timestamp('2019-09-26-18-03-47')
    with pytest.raises(ValueError):filename_timestamp('2019-09-26-18-03-47_unknown')


def test_tar_index_offsets_roundtrip_and_unsafe_names_rejected(tmp_path):
    path=tmp_path/'one.tar'
    with tarfile.open(path,'w') as archive:
        item=tarfile.TarInfo('2020-01-01-00-00-00.jpg');item.size=4
        archive.addfile(item,io.BytesIO(b'test'))
    index=tmp_path/'index.jsonl'
    assert index_tar(path,index) == 1
    row=json.loads(index.read_text());row['tar']=path.name
    assert read_member(tmp_path,row) == b'test'
    with tarfile.open(path,'w') as archive:
        item=tarfile.TarInfo('../escape.jpg');item.size=1
        archive.addfile(item,io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        index_tar(path,index)
