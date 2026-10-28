"""Disagreement-aware DPO loss pieces (no network, no model download)."""

import inspect

import pytest
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer import dpo_trainer as trl_dpo

from prefagent.train.disagreement import (
    _TRL_SIGMOID_LINE, COLUMN, SmoothingCollator, _smoothing, smoothed_logsigmoid,
    smoothing_from_agreement)


def test_smoothing_follows_eq3_and_is_clipped():
    assert smoothing_from_agreement(1.0, 0.5) == 0.0          # unanimous: plain DPO
    assert smoothing_from_agreement(2 / 3, 0.5) == pytest.approx(1 / 6)
    assert smoothing_from_agreement(0.0, 0.5) == 0.5          # fully contested: uniform
    assert smoothing_from_agreement(0.0, 0.2) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="eps_max"):
        smoothing_from_agreement(1.0, 0.6)
    with pytest.raises(ValueError, match="agreement"):
        smoothing_from_agreement(None, 0.5)


def test_smoothed_loss_matches_cdpo_formula():
    x = torch.tensor([-2.0, 0.0, 0.7, 3.0])
    eps = torch.tensor([0.0, 0.1, 0.25, 0.5])
    loss = -smoothed_logsigmoid(x, eps)
    expected = -(1 - eps) * F.logsigmoid(x) - eps * F.logsigmoid(-x)
    torch.testing.assert_close(loss, expected)
    # ε = 0 is exactly standard DPO; and L(ε) = L(0) + ε·x since log σ(x) − log σ(−x) = x.
    torch.testing.assert_close(loss[0], -F.logsigmoid(x[0]))
    torch.testing.assert_close(loss, -F.logsigmoid(x) + eps * x)


def test_smoothing_context_is_scoped_and_checks_shape():
    eps = torch.tensor([0.0, 0.5])
    with _smoothing(eps):
        out = trl_dpo.F.logsigmoid(torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(out, smoothed_logsigmoid(torch.tensor([1.0, 1.0]), eps))
        assert trl_dpo.F.relu is F.relu                        # everything else untouched
        with pytest.raises(RuntimeError, match="per-pair"):
            trl_dpo.F.logsigmoid(torch.zeros(3))
    assert trl_dpo.F is F                                      # restored afterwards


def test_installed_trl_has_the_patched_sigmoid_line():
    # If this fails after a TRL upgrade, the trainer would refuse to start (by design).
    assert _TRL_SIGMOID_LINE in inspect.getsource(DPOTrainer._compute_loss)


def test_collator_adds_per_pair_smoothing():
    c = SmoothingCollator(pad_token_id=0)
    batch = c([{"prompt_ids": [1, 2], "chosen_ids": [3], "rejected_ids": [4, 5], COLUMN: 0.25},
               {"prompt_ids": [1], "chosen_ids": [3, 3], "rejected_ids": [4], COLUMN: 0.0}])
    assert batch[COLUMN].tolist() == [0.25, 0.0]
    assert batch["input_ids"].shape[0] == 4                    # chosen rows, then rejected
