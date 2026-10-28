"""Disagreement-aware DPO: per-pair label smoothing from judge agreement (paper Eq. 3).

    L_i = -(1 - ε_i) log σ(βΔ_i) - ε_i log σ(-βΔ_i),   ε_i = ε_max (1 - a_i)

a_i is the panel agreement stored on each pair (1 = every final verdict ranks chosen
above rejected). Unanimous pairs get ε = 0, the standard DPO loss; contested pairs are
pulled toward a uniform target.

TRL 1.x computes the sigmoid loss inline as `-F.logsigmoid(self.beta * delta_score)`
and ignores `label_smoothing` for it, with no per-example hook. So the trainer below
carries ε through the batch and, for the duration of its own loss call only, gives
TRL's module a `logsigmoid` that applies ε. A source check at construction fails loudly
if a TRL release changes that line, rather than silently training plain DPO.
"""

from __future__ import annotations

import contextlib
import inspect

import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer import dpo_trainer as _trl_dpo
from trl.trainer.dpo_trainer import DataCollatorForPreference

COLUMN = "label_smoothing"
_TRL_SIGMOID_LINE = "-F.logsigmoid(self.beta * delta_score)"


def smoothing_from_agreement(agreement: float, eps_max: float) -> float:
    """ε_i = ε_max (1 - a_i), clipped to [0, 0.5] (0.5 = uniform target)."""
    if not 0.0 <= eps_max <= 0.5:
        raise ValueError(f"eps_max must be in [0, 0.5], got {eps_max}")
    if agreement is None:
        raise ValueError("pair has no agreement score; disagreement-aware DPO needs panel labels")
    return min(0.5, max(0.0, eps_max * (1.0 - float(agreement))))


def smoothed_logsigmoid(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """(1 - ε) log σ(x) + ε log σ(-x), so its negation is the per-pair cDPO loss."""
    return (1.0 - eps) * F.logsigmoid(x) + eps * F.logsigmoid(-x)


class _FunctionalWithSmoothing:
    """torch.nn.functional, except logsigmoid applies per-pair ε to batch-shaped input."""

    def __init__(self, eps: torch.Tensor):
        self._eps = eps

    def logsigmoid(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != self._eps.shape:
            raise RuntimeError(f"unexpected logsigmoid input shape {tuple(x.shape)}; "
                               f"expected per-pair {tuple(self._eps.shape)}")
        return smoothed_logsigmoid(x, self._eps.to(device=x.device, dtype=x.dtype))

    def __getattr__(self, name):
        return getattr(F, name)


@contextlib.contextmanager
def _smoothing(eps: torch.Tensor):
    original = _trl_dpo.F
    _trl_dpo.F = _FunctionalWithSmoothing(eps)
    try:
        yield
    finally:
        _trl_dpo.F = original


class SmoothingCollator(DataCollatorForPreference):
    """TRL's preference collator, plus the per-pair ε as a float tensor."""

    def torch_call(self, examples):
        out = super().torch_call(examples)
        out[COLUMN] = torch.tensor([float(e[COLUMN]) for e in examples], dtype=torch.float32)
        return out


class DisagreementAwareDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        if _TRL_SIGMOID_LINE not in inspect.getsource(DPOTrainer._compute_loss):
            raise RuntimeError("this TRL version computes the sigmoid DPO loss differently; "
                               "update prefagent.train.disagreement before training")
        super().__init__(*args, **kwargs)
        if list(self.loss_types) != ["sigmoid"]:
            raise ValueError(f"disagreement-aware DPO needs loss_type ['sigmoid'], got {self.loss_types}")
        if self.use_weighting or getattr(self, "use_liger_kernel", False):
            raise ValueError("disagreement-aware DPO does not support use_weighting or Liger kernels")
        c = self.data_collator
        self.data_collator = SmoothingCollator(
            pad_token_id=c.pad_token_id, max_length=c.max_length,
            truncation_mode=c.truncation_mode, pad_to_multiple_of=c.pad_to_multiple_of)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if COLUMN not in self._signature_columns:
            self._signature_columns = [*self._signature_columns, COLUMN]

    def _compute_loss(self, model, inputs, return_outputs):
        eps = inputs.pop(COLUMN)
        with _smoothing(eps):
            return super()._compute_loss(model, inputs, return_outputs)
