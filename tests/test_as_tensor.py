"""pinneapple_neural.utils.as_tensor: model output -> prediction Tensor."""
from types import SimpleNamespace

import pytest
import torch

from pinneapple_neural.architectures.autoencoders.base import AEOutput
from pinneapple_neural.architectures.base import ModelOutput
from pinneapple_neural.architectures.neural_operators.base import OperatorOutput
from pinneapple_neural.architectures.pinns.base import PINNOutput
from pinneapple_neural.utils import as_tensor

Y = torch.arange(6.0).reshape(3, 2)


def test_tensor_passthrough_keeps_identity_and_graph():
    x = torch.ones(3, 1, requires_grad=True)
    y = 2 * x
    assert as_tensor(y) is y and as_tensor(y).requires_grad


@pytest.mark.parametrize("out", [
    PINNOutput(y=Y, losses={}), ModelOutput(y=Y), OperatorOutput(y=Y, losses={}, extras={}),
    SimpleNamespace(pred=Y), SimpleNamespace(out=Y), SimpleNamespace(logits=Y), SimpleNamespace(y_hat=Y),
    SimpleNamespace(recon=Y), SimpleNamespace(x=Y), (Y, {"aux": 1}), [Y],
])
def test_wrappers(out):
    assert as_tensor(out) is Y


def test_attribute_priority_and_non_tensor_skipped():
    other = torch.zeros(1)
    assert as_tensor(SimpleNamespace(y=None, pred=Y, x=other)) is Y
    assert as_tensor(SimpleNamespace(y=Y, x=other)) is Y
    assert as_tensor(AEOutput(x_hat=Y, z=other, losses={}, extras={})) is Y


def test_dict_and_field_lists_join_on_last_dim():
    u, v = torch.ones(4), torch.zeros(4)
    assert torch.equal(as_tensor({"u": u, "v": v}), torch.stack([u, v], -1))
    a, b = torch.ones(4, 1), torch.zeros(4, 2)
    assert torch.equal(as_tensor({"a": a, "b": b}), torch.cat([a, b], -1))
    assert torch.equal(as_tensor(PINNOutput(y=[a, b], losses={})), torch.cat([a, b], -1))


def test_custom_attrs_and_errors():
    assert as_tensor(SimpleNamespace(mean=Y), attrs=("mean",)) is Y
    with pytest.raises(TypeError):
        as_tensor(SimpleNamespace(mean=Y))
    with pytest.raises(TypeError):
        as_tensor({"u": 1.0})
    with pytest.raises(TypeError):
        as_tensor(())
