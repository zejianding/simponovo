"""SimPO preference fine-tuning model built on top of PrimeNovo."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .model import Spec2Pep


class PreferenceSpec2Pep(Spec2Pep):
    """PrimeNovo with CTC rewards and a SimPO preference objective.

    This class never calls the beam decoder during training or validation.
    """

    def __init__(
        self,
        num_negatives: int = 2,
        beta: float = 1.0,
        target_margin: float = 0.0,
        positive_ctc_weight: float = 0.1,
        warmup_steps: int = 0,
        total_optimizer_steps: int = 1,
        cosine_min_lr_ratio: float = 0.0,
        enable_inference_decoder: bool = False,
        **kwargs,
    ) -> None:
        if num_negatives < 1:
            raise ValueError("num_negatives must be at least one")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= cosine_min_lr_ratio <= 1.0:
            raise ValueError("cosine_min_lr_ratio must be between zero and one")
        super().__init__(enable_inference_decoder=enable_inference_decoder, **kwargs)
        self.num_negatives = num_negatives
        self.beta = beta
        self.target_margin = target_margin
        self.positive_ctc_weight = positive_ctc_weight
        self.preference_warmup_steps = warmup_steps
        self.preference_total_optimizer_steps = max(total_optimizer_steps, 1)
        self.cosine_min_lr_ratio = cosine_min_lr_ratio
        self._optimizer_step_metric_sums: Dict[str, torch.Tensor] = {}
        self._optimizer_step_micro_batches = 0
        self._last_logged_optimizer_step = 0

    def score_ctc_candidates(
        self,
        logits: torch.Tensor,
        positive_peptides: List[str],
        negative_peptides: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized CTC rewards and per-candidate normalized NLL.

        The returned tensors have shape ``[batch_size, 1 + num_negatives]``.
        ``logits`` is evaluated only once per spectrum; it is expanded only for
        the CTC dynamic-programming calculation.
        """
        batch_size, time_steps, _ = logits.shape
        if len(positive_peptides) != batch_size or len(negative_peptides) != batch_size:
            raise ValueError("Preference batch does not match logits batch size")
        if any(len(row) != self.num_negatives for row in negative_peptides):
            raise ValueError("Each preference row must contain exactly num_negatives negatives")

        all_sequences: List[str] = []
        for positive, negatives in zip(positive_peptides, negative_peptides):
            all_sequences.append(positive)
            all_sequences.extend(negatives)

        token_sequences = [self.decoder.tokenize(sequence) for sequence in all_sequences]
        target_lengths = torch.tensor(
            [tokens.numel() for tokens in token_sequences], device=logits.device, dtype=torch.long
        )
        if torch.any(target_lengths <= 0):
            raise ValueError("Empty peptide target in preference batch")
        if torch.any(target_lengths > time_steps):
            raise ValueError("CTC target is longer than the decoder time dimension")

        targets = torch.cat(token_sequences).to(device=logits.device, dtype=torch.long)
        # CTC is intentionally evaluated in FP32 even under BF16 mixed precision.
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        expanded_log_probs = log_probs.repeat_interleave(self.num_negatives + 1, dim=1)
        input_lengths = torch.full(
            (expanded_log_probs.shape[1],), time_steps, device=logits.device, dtype=torch.long
        )
        nll = F.ctc_loss(
            expanded_log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.decoder.get_blank_idx(),
            reduction="none",
            zero_infinity=False,
        )
        normalized_nll = nll / target_lengths.to(dtype=nll.dtype)
        if not torch.isfinite(normalized_nll).all():
            bad = torch.nonzero(~torch.isfinite(normalized_nll), as_tuple=False).flatten().tolist()
            details = []
            for index in bad[:10]:
                tokens = token_sequences[index]
                repeats = int((tokens[1:] == tokens[:-1]).sum().item()) if tokens.numel() > 1 else 0
                details.append(
                    {
                        "sequence": all_sequences[index],
                        "target_length": int(target_lengths[index]),
                        "required_length": int(target_lengths[index]) + repeats,
                        "time_steps": time_steps,
                    }
                )
            raise FloatingPointError(f"Non-finite CTC candidate score: {details}")
        shape = (batch_size, self.num_negatives + 1)
        normalized_nll = normalized_nll.view(shape)
        return -normalized_nll, normalized_nll

    def compute_preference_losses(
        self,
        logits: torch.Tensor,
        positive_peptides: List[str],
        negative_peptides: List[List[str]],
    ) -> Dict[str, torch.Tensor]:
        rewards, normalized_nll = self.score_ctc_candidates(
            logits, positive_peptides, negative_peptides
        )
        positive_rewards = rewards[:, 0]
        negative_rewards = rewards[:, 1:]
        margins = positive_rewards[:, None] - negative_rewards
        simpo_loss = -F.logsigmoid(self.beta * (margins - self.target_margin)).mean()
        positive_ctc_loss = normalized_nll[:, 0].mean()
        total_loss = simpo_loss + self.positive_ctc_weight * positive_ctc_loss
        return {
            "total_loss": total_loss,
            "simpo_loss": simpo_loss,
            "positive_ctc_loss": positive_ctc_loss,
            "positive_rewards": positive_rewards,
            "negative_rewards": negative_rewards,
            "margins": margins,
        }

    def _shared_step(self, batch: Dict[str, object], stage: str) -> torch.Tensor:
        spectra = batch["spectra"]
        precursors = batch["precursors"]
        positive_peptides = batch["positive_peptides"]
        negative_peptides = batch["negative_peptides"]
        if not isinstance(spectra, torch.Tensor) or not isinstance(precursors, torch.Tensor):
            raise TypeError("Preference batch is missing spectrum tensors")
        logits, _, _ = self._forward_step(spectra, precursors, positive_peptides)
        losses = self.compute_preference_losses(logits, positive_peptides, negative_peptides)
        metrics = {
            "total_loss": losses["total_loss"],
            "simpo_loss": losses["simpo_loss"],
            "positive_ctc_loss": losses["positive_ctc_loss"],
            "positive_reward": losses["positive_rewards"].mean(),
            "negative_reward_mean": losses["negative_rewards"].mean(),
            "reward_margin_mean": losses["margins"].mean(),
            "preference_accuracy": (losses["margins"] > 0).float().mean(),
        }
        if stage == "train":
            self._accumulate_optimizer_step_metrics(metrics)
            epoch_log_kwargs = dict(
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=spectra.shape[0],
            )
            for name, value in metrics.items():
                self.log(f"train/{name}_epoch", value, **epoch_log_kwargs)
        else:
            val_log_kwargs = dict(
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=spectra.shape[0],
            )
            for name, value in metrics.items():
                self.log(f"val/{name}", value, **val_log_kwargs)
        return losses["total_loss"]

    def _accumulate_optimizer_step_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        """Accumulate detached micro-batch metrics until the next optimizer update."""
        for name, value in metrics.items():
            detached = value.detach().float()
            if name not in self._optimizer_step_metric_sums:
                self._optimizer_step_metric_sums[name] = detached.clone()
            else:
                self._optimizer_step_metric_sums[name] = (
                    self._optimizer_step_metric_sums[name] + detached
                )
        self._optimizer_step_micro_batches += 1

    def _reset_optimizer_step_metrics(self) -> None:
        self._optimizer_step_metric_sums.clear()
        self._optimizer_step_micro_batches = 0

    def training_step(self, batch: Dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        """Disable the legacy PrimeNovo CELoss history hook."""

    def on_validation_epoch_end(self) -> None:
        """Disable legacy decode-metric aggregation during Windows training."""

    def on_train_start(self) -> None:
        """Initialize accumulation-aware logging, including after checkpoint resume."""
        self._reset_optimizer_step_metrics()
        self._last_logged_optimizer_step = int(self.trainer.global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log exactly one accumulated metric set for each optimizer update."""
        optimizer_step = int(self.trainer.global_step)
        if optimizer_step <= self._last_logged_optimizer_step:
            return
        if self._optimizer_step_micro_batches <= 0:
            raise RuntimeError("Optimizer step completed without accumulated train metrics")

        accumulation_steps = int(self.trainer.accumulate_grad_batches)
        if accumulation_steps <= 0:
            raise RuntimeError("accumulate_grad_batches must be positive")
        loss_names = {"total_loss", "simpo_loss", "positive_ctc_loss"}
        for name, metric_sum in self._optimizer_step_metric_sums.items():
            if name in loss_names:
                # Lightning divides every micro-batch loss by the configured
                # accumulation factor before backward.  Summing those scaled
                # losses reproduces the objective that produced this update,
                # including the final incomplete accumulation window.
                value = metric_sum / accumulation_steps
            else:
                value = metric_sum / self._optimizer_step_micro_batches
            self.log(
                f"train/{name}_step",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                logger=True,
                prog_bar=name == "total_loss",
            )
        self._last_logged_optimizer_step = optimizer_step
        self._reset_optimizer_step_metrics()

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None) -> None:
        """TensorBoard-safe replacement for the legacy dict-style logger hook."""
        total = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                total = total + parameter.grad.detach().norm(2).pow(2)
        self.log(
            "train/grad_norm",
            total.sqrt(),
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs)

        def lr_lambda(step: int) -> float:
            warmup_steps = self.preference_warmup_steps
            total_steps = self.preference_total_optimizer_steps
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(self.preference_warmup_steps)
            decay_steps = total_steps - warmup_steps
            if decay_steps <= 1:
                return self.cosine_min_lr_ratio
            progress = min(max((step - warmup_steps) / float(decay_steps - 1), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cosine_min_lr_ratio + (1.0 - self.cosine_min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
