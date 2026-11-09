"""Physical-label and leakage checks for the imaging starter intake."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from ml.data_pipeline.import_public_imaging import aperture_hfr, moment_shape, prepare_synthetic, render_profile
from ml.data_pipeline.split_registry import load_registered_splits
from ml.labels.psf_measure import require_sep


@pytest.fixture
def sep_required():
    """SEP-backed tests fail with "missing prerequisite: sep==1.4.1" rather than skip."""
    return require_sep()


def test_fits_intake_preserves_quality_arrays_and_rejects_masked_cores(tmp_path, sep_required):
    from astropy.io import fits
    from ml.data_pipeline.import_public_imaging import prepare_hst
    product='fixture123'
    output=tmp_path/'diagnostic';original=output/'originals'/f'{product}_flt.fits'
    original.parent.mkdir(parents=True)
    y,x=np.indices((256,256));image=np.full((256,256),100,dtype=np.float32)
    for cx,cy in ((55,55),(180,55),(55,180),(180,180)):
        image+=2000*np.exp(-((x-cx)**2+(y-cy)**2)/8).astype(np.float32)
    dq=np.zeros(image.shape,np.uint16)
    dq[55,55]=4;dq[55,180]=256;dq[180,65]=4
    primary=fits.PrimaryHDU()
    for k,v in {'INSTRUME':'ACS','ROOTNAME':product,'DATE-OBS':'2020-01-01','FILTER1':'F606W','FILTER2':'CLEAR2L'}.items():primary.header[k]=v
    sci=fits.ImageHDU(image,name='SCI');sci.header['EXTVER']=1;sci.header['BUNIT']='ELECTRONS'
    err=fits.ImageHDU(np.sqrt(image+25),name='ERR');err.header['EXTVER']=1
    flags=fits.ImageHDU(dq,name='DQ');flags.header['EXTVER']=1
    fits.HDUList([primary,sci,err,flags]).writeto(original,checksum=True)
    before=original.read_bytes()
    manifest=prepare_hst(tmp_path,product=product,dataset_name='diagnostic',expected_size=original.stat().st_size)
    rows=[json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    accepted=[r for r in rows if r['fqe_quality_accepted']]
    assert len(accepted)==2
    assert {r['label_quality_tier'] for r in accepted}=={'strict_unmasked_aperture','mask_aware_experimental'}
    assert all(r['filter_name']=='F606W' and not r['training_eligible'] for r in rows)
    assert any('saturated_dq' in r['rejections'] for r in rows)
    assert any('quality_masked_core_or_wings' in r['rejections'] for r in rows)
    assert original.read_bytes()==before
    with np.load(output/'frame.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['science_electrons'],image)
        np.testing.assert_array_equal(data['dq'],dq)
        background = data['sep_background_estimate_electrons']
    assert manifest['fqe_quality_accepted']==2
    # SEP provenance (BGE-20). The background is the same SEP call as before
    # (library-default mesh, DQ/finite/ERR mask); noise stays the pipeline ERR.
    import sep
    from ml.labels.psf_measure import config_sha256
    mask=(dq!=0)|~np.isfinite(image)
    cleaned=image.copy();cleaned[mask]=float(np.median(image[~mask]))
    np.testing.assert_array_equal(background,sep.Background(cleaned,mask=mask).back())
    assert manifest['sep_version']=='1.4.1'
    assert manifest['noise_source']=='pipeline_error'
    assert manifest['background_config']['method']=='sep_spatial_mesh'
    assert (manifest['background_config']['bw'],manifest['background_config']['fw'])==(64,3)
    assert manifest['background_config_sha256']==config_sha256(manifest['background_config'])
    detection=manifest['detection_config']
    assert (detection['thresh_sigma'],detection['filter_type'],detection['noise'])==(6.0,'matched','pipeline_error_map')
    assert detection['pixstack']>=image.size
    assert manifest['detection_config_sha256']==config_sha256(detection)
    assert manifest['background_provenance']['units']=='electrons'
    assert all(r['noise_source']=='pipeline_error' and r['detection_config_sha256']==manifest['detection_config_sha256'] for r in rows)
    assert not manifest['training_eligible'] and manifest['split']=='test'
    assert any('do not calibrate Canon' in item for item in manifest['limitations'])


def _gaussian(**changes):
    params = dict(center=[31.5, 31.5], profile="gaussian", width_px=2.0,
                  axis_ratio=1.0, optical_pa_rad=0.0, motion="none",
                  trail_length_px=0.0, trail_pa_image_rad=0.0,
                  moffat_beta=3.0, flux_electrons=10000.0)
    params.update(changes)
    return params


def test_aperture_hfr_insists_on_the_pinned_sep():
    import sys
    import types
    from unittest import mock
    from ml.labels.psf_measure import MissingPrerequisite

    fake = types.ModuleType("sep")
    fake.__version__ = "1.2.1"
    with mock.patch.dict(sys.modules, {"sep": fake}):
        with pytest.raises(MissingPrerequisite, match="missing prerequisite: sep==1.4.1"):
            aperture_hfr(render_profile(64, _gaussian()), (31.5, 31.5), 24)


def test_reference_hfr_is_finite_pixel_not_width_parameter(sep_required):
    image = render_profile(64, _gaussian(), oversample=5)
    hfr = aperture_hfr(image, (31.5, 31.5), 24)
    assert abs(image.sum()-10000) < 0.1
    assert abs(hfr - 2*np.sqrt(2*np.log(2))) < 0.18
    assert hfr != 2.0
    wide = render_profile(64, _gaussian(width_px=4), oversample=5)
    assert aperture_hfr(wide, (31.5, 31.5), 24) > hfr
    assert aperture_hfr(np.zeros((64, 64)), (31.5, 31.5), 24) is None


def test_motion_shape_uses_image_coordinates_and_round_angles_undefined():
    round_image = render_profile(64, _gaussian())
    ell, angle = moment_shape(round_image)
    assert ell < 1e-6
    assert angle is None
    for orientation in (0, np.pi/2):
        trail = render_profile(64, _gaussian(motion="linear", trail_length_px=10, trail_pa_image_rad=orientation))
        ell, angle = moment_shape(trail)
        assert ell > 0.35
        assert abs(np.cos(2*(angle-orientation))-1) < 0.01


def test_edge_truncation_preserves_lost_flux():
    centered = render_profile(64, _gaussian())
    edge = render_profile(64, _gaussian(center=[0.0, 31.5]))
    assert edge.sum() < centered.sum()*0.65
    assert edge.sum() > centered.sum()*0.45


def test_prepared_schema_splits_masks_and_no_fake_focus_runs(sep_required):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = prepare_synthetic(root, seed=42, families=32, variants=2, scene_count=2)
        output = root / "stellar_sim_v1"
        data = np.load(output / "fqe.npz", allow_pickle=False)
        assert data["crops"].dtype == np.float32
        assert data["crops"].shape[1:] == (64, 64)
        assert np.isfinite(data["crops"]).all()
        assert data["crops"].min() >= 0 and data["crops"].max() <= 1
        assert np.allclose(np.exp(data["log_hfr"]), data["hfr_px"])
        assert len(set(data["run_id"])) == len(data["run_id"])
        assert not data["ranking_eligible"].any()
        assert "position" not in data.files
        assert not np.allclose(data["measured_hfr_px"], data["reference_hfr_px"])
        splits = load_registered_splits(str(output / "fqe.splits.json"), str(output / "fqe.npz"), data["split_group"])
        for name, label in (("train", "train"), ("val", "validation"), ("test", "test")):
            assert np.all(data["split"][splits[name]] == label)
        rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
        by_family = {}
        for row in rows:
            by_family.setdefault(row["ancestor_id"], set()).add(row["split"])
            assert row["signed_focus_offset"] is None
            assert row["focuser_position"] is None
        assert all(len(value) == 1 for value in by_family.values())
        assert manifest["observations"] == 64
        assert manifest["sep_version"] == "1.4.1"
        obs = np.load(output / "observations.npz", allow_pickle=False)
        assert obs["pixel_flags"].dtype == np.uint8
        assert obs["raw_crops"].shape == (64, 64, 64)
        ted = np.load(output / "ted.npz", allow_pickle=False)
        assert ted["crops"].shape == (64, 32, 32)
        assert np.isnan(ted["reference_pa_image_rad"][~ted["reference_angle_valid"]]).all()
        bge = np.load(output / "bge.npz", allow_pickle=False)
        assert bge["images"].shape == (2, 128, 128)
        assert (bge["variance_truth_preclip"] > 0).all()
        assert bge["background_truth"].min() > 500


def test_same_seed_same_arrays_and_ancestry(sep_required):
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        prepare_synthetic(Path(first), seed=83, families=32, variants=1, scene_count=0)
        prepare_synthetic(Path(second), seed=83, families=32, variants=1, scene_count=0)
        a = np.load(Path(first)/"stellar_sim_v1/fqe.npz", allow_pickle=False)
        b = np.load(Path(second)/"stellar_sim_v1/fqe.npz", allow_pickle=False)
        for key in a.files:
            assert np.array_equal(a[key], b[key], equal_nan=True) if a[key].dtype.kind == "f" else np.array_equal(a[key], b[key])
