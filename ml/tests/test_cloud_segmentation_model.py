"""COC tensor, partial-label and distillation contract tests."""
import json

import pytest

torch = pytest.importorskip("torch")

from ml.models.cloud_classifier import CloudClassifier, CloudSegmenter, variant_config
from ml.models.cloud_losses import masked_cross_entropy, masked_teacher_kl


@pytest.mark.parametrize("variant", ["tiny", "small"])
def test_segmenter_keeps_odd_spatial_geometry_and_backpropagates(variant):
    model = CloudSegmenter(variant)
    output = model(torch.rand(2, 3, 17, 25))
    assert output.shape == (2, 3, 17, 25)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0


def test_variants_share_ontology_and_differ_in_capacity():
    tiny = CloudSegmenter("tiny")
    small = CloudSegmenter("small")
    assert sum(p.numel() for p in tiny.parameters()) < sum(p.numel() for p in small.parameters())
    config = json.loads(json.dumps(variant_config("tiny")))
    assert config["class_names"] == ["sky", "cloud", "contamination"]
    assert config["ignore_index"] == 255
    assert config["pretrained"] is False
    with pytest.raises(ValueError, match="variant"):
        CloudSegmenter("large")
    with pytest.raises(NotImplementedError, match="legacy five-class"):
        CloudClassifier()


@pytest.mark.parametrize(
    "image",
    [torch.zeros(1, 1, 16, 16), torch.zeros(1, 3, 15, 16),
     torch.full((1, 3, 16, 16), float("nan")), torch.full((1, 3, 16, 16), 255.0),
     torch.zeros(1, 3, 16, 16, dtype=torch.uint8)],
)
def test_segmenter_rejects_unsupported_input(image):
    with pytest.raises(ValueError):
        CloudSegmenter()(image)


def test_partial_ce_ignores_unannotated_logits_and_their_gradients():
    logits = torch.tensor([[[[2.0, 1.0]], [[0.0, 2.0]], [[-1.0, 3.0]]]], requires_grad=True)
    targets = torch.tensor([[[0, 255]]])
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = masked_cross_entropy(logits, targets, weights)
    expected = torch.nn.functional.cross_entropy(logits[:, :, :, :1], targets[:, :, :1], weight=weights)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, :, 1]) == 0
    assert logits.grad[:, :, :, 0].abs().sum() > 0


def test_all_ignored_ce_is_finite_differentiable_zero():
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    targets = torch.full((2, 4, 5), 255, dtype=torch.long)
    loss = masked_cross_entropy(logits, targets)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_all_ignored_ce_does_not_overflow_when_finite_logits_sum_would():
    logits = torch.full((1, 3, 2, 2), 1e38, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.full((1, 2, 2), 255, dtype=torch.long))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("bad_label", [-1, 3, 254, 256])
def test_ce_rejects_unsupported_labels(bad_label):
    with pytest.raises(ValueError, match="only sky"):
        masked_cross_entropy(torch.zeros(1, 3, 2, 2), torch.full((1, 2, 2), bad_label))


def test_ce_rejects_nonfinite_logits_and_invalid_weights():
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="finite"):
        masked_cross_entropy(torch.full((1, 3, 2, 2), float("inf")), labels)
    with pytest.raises(ValueError, match="positive"):
        masked_cross_entropy(torch.zeros(1, 3, 2, 2), labels, torch.tensor([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="int64"):
        masked_cross_entropy(torch.zeros(1, 3, 2, 2), labels.float())


def test_teacher_kl_only_uses_confident_content_and_detaches_teacher():
    student = torch.zeros(1, 3, 4, 4, requires_grad=True)
    teacher = torch.zeros(1, 3, 2, 2, requires_grad=True)
    with torch.no_grad():
        teacher[0, 1, 0, 0] = 6
        teacher[0, 2, 0, 1] = 6
    content = torch.tensor([[[True, False], [True, True]]])
    loss, count = masked_teacher_kl(student, teacher, content)
    assert count == 1  # Confident outside-content pixel is excluded.
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_teacher_kl_zero_selected_is_finite_differentiable_zero():
    student = torch.zeros(1, 3, 4, 4, requires_grad=True)
    teacher = torch.zeros(1, 3, 2, 2)
    loss, count = masked_teacher_kl(student, teacher, torch.ones(1, 2, 2, dtype=torch.bool))
    assert count == 0 and loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0


def test_teacher_kl_matches_manual_temperature_scaled_selected_mean():
    student = torch.tensor([[[[1.0]], [[2.0]], [[-1.0]]]], requires_grad=True)
    teacher = torch.tensor([[[[3.0]], [[0.0]], [[-2.0]]]])
    temperature = 2.0
    loss, count = masked_teacher_kl(student, teacher, torch.ones(1, 1, 1, dtype=torch.bool),
                                  temperature=temperature, confidence_threshold=0)
    target = torch.softmax(teacher / temperature, dim=1)
    expected = (target * (target.log() - torch.log_softmax(student / temperature, dim=1))).sum() * 4
    torch.testing.assert_close(loss, expected)
    assert count == 1


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": float("nan")},
                                   {"confidence_threshold": 1.1}])
def test_teacher_kl_rejects_invalid_selection_parameters(kwargs):
    with pytest.raises(ValueError):
        masked_teacher_kl(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 2, 2),
                          torch.ones(1, 2, 2, dtype=torch.bool), **kwargs)


def test_teacher_kl_rejects_nonboolean_content_and_nonfinite_teacher():
    with pytest.raises(ValueError, match="boolean"):
        masked_teacher_kl(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 2, 2), torch.ones(1, 2, 2))
    with pytest.raises(ValueError, match="finite"):
        masked_teacher_kl(torch.zeros(1, 3, 4, 4), torch.full((1, 3, 2, 2), float("nan")),
                          torch.ones(1, 2, 2, dtype=torch.bool))
