"""Tests for combining StepOutput values."""

from typing import Any

import pytest
import torch

from torch_batteries import StepOutput


class TestStepOutputAdd:
    """Test cases for StepOutput addition."""

    def test_add_sums_losses_and_merges_keyed_values(self) -> None:
        """Test that losses are summed and keyed values are merged."""
        logits, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
        price, true_price = torch.randn(4), torch.randn(4)
        classification = StepOutput(
            loss=torch.tensor(1.0),
            predictions={"classification": logits},
            targets={"classification": labels},
            metrics={"confidence": 0.5},
        )
        regression = StepOutput(
            loss=torch.tensor(2.0),
            predictions={"regression": price},
            targets={"regression": true_price},
        )

        combined = classification + regression

        assert combined.loss is not None
        assert combined.loss.item() == pytest.approx(3.0)
        assert combined.predictions == {"classification": logits, "regression": price}
        assert combined.targets == {"classification": labels, "regression": true_price}
        assert combined.metrics == {"confidence": 0.5}

    def test_add_keeps_one_sided_tensor_values(self) -> None:
        """Test that unkeyed tensors are kept when only one side sets them."""
        predictions, targets = torch.randn(2), torch.randn(2)
        supervised = StepOutput(
            loss=torch.tensor(1.0), predictions=predictions, targets=targets
        )
        penalty = StepOutput(loss=torch.tensor(0.1), metrics={"penalty": 0.1})

        combined = supervised + penalty

        assert combined.predictions is predictions
        assert combined.targets is targets
        assert combined.metrics == {"penalty": 0.1}

    def test_add_preserves_gradient_flow(self) -> None:
        """Test that the summed loss backpropagates to both terms."""
        weight = torch.tensor(1.0, requires_grad=True)
        combined = StepOutput(loss=weight * 2) + StepOutput(loss=weight * 3)

        assert combined.loss is not None
        combined.loss.backward()

        assert weight.grad is not None
        assert weight.grad.item() == pytest.approx(5.0)

    def test_sum_combines_outputs(self) -> None:
        """Test that built-in sum combines several outputs."""
        outputs = [
            StepOutput(loss=torch.tensor(float(index)), metrics={f"m{index}": index})
            for index in range(3)
        ]

        combined = sum(outputs)

        assert isinstance(combined, StepOutput)
        assert combined.loss is not None
        assert combined.loss.item() == pytest.approx(3.0)
        assert combined.metrics == {"m0": 0, "m1": 1, "m2": 2}

    def test_add_skips_missing_losses(self) -> None:
        """Test that a missing loss does not count as zero or erase the other."""
        loss = torch.tensor(1.5)
        with_loss = StepOutput(loss=loss)
        without_loss = StepOutput(metrics={"success_rate": 0.5})

        assert (with_loss + without_loss).loss is loss
        assert (without_loss + StepOutput()).loss is None

    def test_add_does_not_mutate_operands(self) -> None:
        """Test that addition returns a new output."""
        left = StepOutput(loss=torch.tensor(1.0), metrics={"a": 1.0})
        right = StepOutput(loss=torch.tensor(2.0), metrics={"b": 2.0})

        _ = left + right

        assert left.metrics == {"a": 1.0}
        assert right.metrics == {"b": 2.0}

    @pytest.mark.parametrize("field_name", ["predictions", "targets", "metrics"])
    def test_add_rejects_duplicate_keys(self, field_name: str) -> None:
        """Test that overlapping keys fail instead of silently overwriting."""
        value: dict[str, Any] = {"shared": torch.tensor(1.0)}
        left = StepOutput(loss=torch.tensor(1.0), **{field_name: value})
        right = StepOutput(loss=torch.tensor(1.0), **{field_name: value})

        with pytest.raises(ValueError, match=f"duplicate {field_name} keys"):
            _ = left + right

    def test_add_rejects_two_unkeyed_tensors(self) -> None:
        """Test that two unkeyed prediction tensors are ambiguous."""
        left = StepOutput(loss=torch.tensor(1.0), predictions=torch.randn(2))
        right = StepOutput(loss=torch.tensor(1.0), predictions=torch.randn(2))

        with pytest.raises(TypeError, match="predictions are both set"):
            _ = left + right

    def test_add_rejects_non_step_outputs(self) -> None:
        """Test that unsupported operands raise TypeError."""
        output = StepOutput(loss=torch.tensor(1.0))

        with pytest.raises(TypeError):
            _ = output + torch.tensor(1.0)
        with pytest.raises(TypeError):
            _ = 1 + output
