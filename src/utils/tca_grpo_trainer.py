"""Multi-positive TCA-inspired T5 GRPO (independent implementation).

Method reference: Zhao et al., arXiv:2608.12049; author implementation:
https://github.com/colinzhaoxp/TCA-RL4GR (absolute trajectory + global hit reward).
Gold states are computed on demand from an immutable SFT reference. The bounded
CPU cache, multi-positive path choice, reward-variance floor, and optional
all-wrong supervision are adaptations, not claims of exact paper reproduction.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
import random
from typing import Any

import torch
from torch.nn import functional as F
from accelerate.utils import convert_to_fp32

from utils.grpo_trainer import GRPOSettings, Seq2SeqGRPOTrainer, completion_mask, grpo_loss, token_logps


@dataclass
class TCASettings(GRPOSettings):
    objective: str = "tca_grpo"
    hit_bonus: float = 2.
    overflow_penalty: float = -.5
    reward_std_floor: float = .001
    gold_candidates_per_query: int = 2
    gold_cache_mb: float = 128.
    reference_batch_size: int = 16
    all_zero_ranking_weight: float = .1
    sft_anchor_weight: float = .1
    negatives_per_query: int = 4
    trajectory_alignment: str = "pre_token"

    def __post_init__(self):
        super().__post_init__()
        values = (self.hit_bonus, self.overflow_penalty, self.reward_std_floor, self.gold_cache_mb,
                  self.all_zero_ranking_weight, self.sft_anchor_weight)
        if not all(isfinite(value) for value in values):
            raise ValueError("TCA settings must be finite")
        if self.hit_bonus < 0 or self.overflow_penalty > 0 or min(values[2:]) < 0:
            raise ValueError("Require nonnegative bonus, floor, cache and weights, and nonpositive overflow penalty")
        if min(self.gold_candidates_per_query, self.reference_batch_size, self.negatives_per_query) < 1:
            raise ValueError("Gold, reference and negative budgets must be positive")
        if self.ranking_loss_weight:
            raise ValueError("Use all_zero_ranking_weight for TCA, not the legacy generated-candidate ranking loss")
        if self.trajectory_alignment not in {"pre_token", "post_token"}:
            raise ValueError("trajectory_alignment must be pre_token or post_token")

    @property
    def reward_description(self):
        return "position-aligned decoder cosine on content tokens, overflow penalty, all-positive-set exact-hit bonus"


class GoldTrajectoryCache:
    """Per-process CPU LRU, bounded by tensor storage rather than entry count."""
    def __init__(self, max_mb=128.):
        self.limit = int(max_mb * 1024**2)
        self.bytes = 0
        self.entries = OrderedDict()

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        size = value.numel() * 4
        if not self.limit or size > self.limit:
            return
        if key in self.entries:
            self.bytes -= self.entries.pop(key).numel() * 4
        while self.entries and self.bytes + size > self.limit:
            self.bytes -= self.entries.popitem(last=False)[1].numel() * 4
        self.entries[key] = value.detach().float().cpu().clone()
        self.bytes += size


def padded_sequences(tokens, index, device):
    sequences = torch.full((len(tokens), max(map(len, tokens))), index.pad_id, dtype=torch.long, device=device)
    for i, row in enumerate(tokens):
        sequences[i, :len(row)] = torch.tensor(row, device=device)
    return sequences


@torch.no_grad()
def gold_trajectories(reference, inputs, gold_targets, index, settings, cache):
    """Use decoder-only final states, avoiding LM-head logits for gold paths.

    State t is produced by [decoder_start, gold[:t]] and predicts gold[t].
    EOS is excluded from the trajectory similarity, matching the author's
    absolute/global-hit reward function. Query tokens are part of every key.
    """
    device = inputs["input_ids"].device
    features, missing = {}, []
    for owner, targets in enumerate(gold_targets):
        query = tuple(inputs["input_ids"][owner][inputs["attention_mask"][owner].bool()].tolist())
        for target in targets:
            key = (query, target, settings.trajectory_alignment)
            value = cache.get(key)
            if value is None:
                missing.append((owner, target, key))
            else:
                features[(owner, target)] = value.to(device)
    for offset in range(0, len(missing), settings.reference_batch_size):
        batch = missing[offset:offset + settings.reference_batch_size]
        owners = torch.tensor([item[0] for item in batch], device=device)
        source = {name: inputs[name].index_select(0, owners) for name in ("input_ids", "attention_mask")}
        rows = [[index.start_id, *index.target_to_sequence[target]] for _, target, _ in batch]
        sequences = padded_sequences(rows, index, device)
        mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
        encoder = reference.get_encoder()(**source, return_dict=True)
        decoder = reference.get_decoder()(
            input_ids=sequences[:, :-1], attention_mask=torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), 1),
            encoder_hidden_states=encoder.last_hidden_state, encoder_attention_mask=source["attention_mask"],
            use_cache=False, return_dict=True)
        for i, (owner, target, key) in enumerate(batch):
            content_length = len(index.target_to_sequence[target]) - 1
            shift = int(settings.trajectory_alignment == "post_token")
            value = decoder.last_hidden_state[i, shift:shift + content_length].detach().float().clone()
            features[(owner, target)] = value
            cache.put(key, value)
        del encoder, decoder
    return features


@torch.no_grad()
def trajectory_rewards(hidden, sequences, positives, gold_targets, features, index, settings):
    """Choose one whole gold path per completion; do not splice token-wise maxima."""
    mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
    rewards = hidden.new_zeros(mask.shape, dtype=torch.float32)
    exact = index.rewards(sequences, positives, settings.num_generations)
    canonical = [index.canonical(row) for row in sequences.detach().cpu().tolist()]
    for i in range(len(sequences)):
        owner = i // settings.num_generations
        length = int(mask[i].sum())
        content = length - int(index.eos_id in canonical[i])
        target = index.sequence_to_target.get(canonical[i])
        shift = int(settings.trajectory_alignment == "post_token")
        if content + shift > hidden.shape[1]:
            raise ValueError("Post-token rewards need a hidden state after every content token; completion must include EOS")
        paths = [target] if exact[i] else gold_targets[owner]
        best, best_score = None, float("-inf")
        for gold in paths:
            gold_hidden = features[(owner, gold)]
            overlap = min(content, len(gold_hidden))
            candidate = rewards.new_zeros(mask.shape[1])
            if overlap:
                candidate[:overlap] = F.cosine_similarity(hidden[i, shift:shift + overlap].detach().float(),
                                                         gold_hidden[:overlap].float(), dim=-1).clamp(-1, 1)
            if content > len(gold_hidden):
                candidate[len(gold_hidden):content] = settings.overflow_penalty
            score = float(candidate[:content].sum() / max(content, 1))
            if score > best_score:
                best, best_score = candidate, score
        rewards[i] = best + settings.hit_bonus * exact[i] * mask[i]
    return rewards * mask, exact


def token_group_advantages(rewards, mask, group_size, std_floor=.001):
    """Normalize across valid candidates at each depth; padding never votes."""
    grouped = rewards.detach().float().reshape(-1, group_size, rewards.shape[1])
    active = mask.reshape_as(grouped).bool()
    counts = active.sum(1, keepdim=True)
    means = (grouped * active).sum(1, keepdim=True) / counts.clamp_min(1)
    centered = (grouped - means) * active
    std = (centered.square().sum(1, keepdim=True) / counts.clamp_min(1)).sqrt()
    informative = (counts > 1) & (std > max(std_floor, 1e-8))
    advantages = centered / std.clamp_min(max(std_floor, 1e-8))
    return (advantages * informative).reshape_as(rewards), informative.squeeze(1)


def compute_tca_objective(model, reference, inputs, generated, index, settings, cache):
    """Shared by training and the GPU capacity probe; all gradients stay in policy."""
    batch = len(inputs["positive_targets"])
    device = generated.device
    rng = random.Random(torch.randint(0, 2**31, ()).item())
    canonical = [index.canonical(row) for row in generated.detach().cpu().tolist()]
    exact = index.rewards(generated, inputs["positive_targets"], settings.num_generations)
    selected, reward_gold = [], []
    for i, positives in enumerate(inputs["positive_targets"]):
        pool = sorted(set(positives))
        rng.shuffle(pool)
        selected.append(pool[:settings.gold_candidates_per_query])
        hits = [index.sequence_to_target[canonical[j]] for j in range(i * settings.num_generations, (i + 1) * settings.num_generations)
                if exact[j]]
        reward_gold.append(list(dict.fromkeys(selected[-1] + hits)))
    features = gold_trajectories(reference, inputs, reward_gold, index, settings, cache)
    # At most G+K policy rows/query: sampled scores are reused as ranking
    # negatives instead of constructing a second negative-scoring graph.
    token_rows = generated.detach().cpu().tolist()
    owners = list(range(batch))
    owners = [owner for owner in owners for _ in range(settings.num_generations)]
    supervised_rows = []
    if settings.sft_anchor_weight or settings.all_zero_ranking_weight:
        for owner, targets in enumerate(selected):
            positions = []
            for target in targets:
                positions.append(len(token_rows))
                token_rows.append([index.start_id, *index.target_to_sequence[target]])
                owners.append(owner)
            supervised_rows.append(positions)
    sequences = padded_sequences(token_rows, index, device)
    full_mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
    owner_ids = torch.tensor(owners, device=device)
    repeated = {name: inputs[name].index_select(0, owner_ids) for name in ("input_ids", "attention_mask")}
    generated_count = len(generated)
    reference_logps = None
    if settings.beta:
        with torch.no_grad():
            ref_mask = full_mask[:generated_count]
            ref_outputs = convert_to_fp32(reference(
                **{name: value[:generated_count] for name, value in repeated.items()},
                decoder_input_ids=sequences[:generated_count, :-1],
                decoder_attention_mask=torch.cat((torch.ones_like(ref_mask[:, :1]), ref_mask[:, :-1]), 1), use_cache=False))
            reference_logps = token_logps(ref_outputs.logits, index.score_layout(sequences[:generated_count]),
                                         settings.scoring_temperature, settings.score_normalization)
            del ref_outputs
    outputs = convert_to_fp32(model(**repeated, decoder_input_ids=sequences[:, :-1],
                                    decoder_attention_mask=torch.cat((torch.ones_like(full_mask[:, :1]), full_mask[:, :-1]), 1),
                                    use_cache=False, output_hidden_states=True))
    all_logps = token_logps(outputs.logits, index.score_layout(sequences), settings.scoring_temperature, settings.score_normalization)
    # Hidden states/bonuses are reward observations, never a differentiable
    # cosine objective through either policy or frozen reference.
    rewards, exact = trajectory_rewards(outputs.decoder_hidden_states[-1][:generated_count].detach(),
                                        sequences[:generated_count], inputs["positive_targets"], reward_gold,
                                        features, index, settings)
    del features
    mask = full_mask[:generated_count]
    advantages, informative = token_group_advantages(rewards, mask, settings.num_generations, settings.reward_std_floor)
    logps = all_logps[:generated_count]
    objective, kl = grpo_loss(logps, logps.detach(), advantages, mask, settings.epsilon, settings.beta, reference_logps)
    ranking_logps = all_logps if settings.scoring_temperature == 1. else token_logps(
        outputs.logits, index.score_layout(sequences), 1., settings.score_normalization)
    scores = (ranking_logps * full_mask).sum(1)
    rank_terms, sft_terms = [], []
    for owner, positions in enumerate(supervised_rows):
        ids = torch.tensor(positions, device=device)
        if settings.sft_anchor_weight:
            labels = sequences[ids, 1:].masked_fill(~full_mask[ids], -100)
            terms = F.cross_entropy(outputs.logits[ids].reshape(-1, outputs.logits.shape[-1]), labels.reshape(-1),
                                    reduction="none").reshape_as(labels)
            sft_terms.append((terms.sum(1) / full_mask[ids].sum(1).clamp_min(1)).mean())
        start, end = owner * settings.num_generations, (owner + 1) * settings.num_generations
        if settings.all_zero_ranking_weight and not exact[start:end].any():
            order = sorted(range(start, end), key=lambda j: float(scores[j].detach()), reverse=True)
            unique = {}
            for j in order:
                unique.setdefault(canonical[j], j)
            negatives = list(unique.values())[:settings.negatives_per_query]
            differences = scores[negatives][None, :] - scores[ids][:, None]
            rank_terms.append(torch.logsumexp(torch.cat((torch.zeros_like(differences[:, :1]), differences), 1), 1).mean())
        else:
            rank_terms.append(scores[ids].sum() * 0.)
    rank_loss = torch.stack(rank_terms).mean() if rank_terms else objective * 0.
    sft_loss = torch.stack(sft_terms).mean() if sft_terms else objective * 0.
    loss = objective + settings.all_zero_ranking_weight * rank_loss + settings.sft_anchor_weight * sft_loss
    del outputs
    grouped_exact = exact.reshape(batch, settings.num_generations).mean(1)
    duplicates = [1 - len(set(canonical[i * settings.num_generations:(i + 1) * settings.num_generations])) / settings.num_generations
                  for i in range(batch)]
    stats = {"total_loss": loss.detach(), "policy_loss": objective.detach() - settings.beta * kl.detach(),
             "kl": kl.detach(), "kl_loss": settings.beta * kl.detach(),
             "token_reward": (rewards * mask).sum() / mask.sum().clamp_min(1),
             "advantage_abs_mean": (advantages.abs() * mask).sum() / mask.sum().clamp_min(1),
             "informative_depth_fraction": informative.sum() / mask.reshape(batch, settings.num_generations, -1).any(1).sum().clamp_min(1),
             "exact_reward": exact.mean(), "all_zero_groups": grouped_exact.eq(0).float().mean(),
             "all_one_groups": grouped_exact.eq(1).float().mean(),
             "mixed_reward_groups": ((grouped_exact > 0) & (grouped_exact < 1)).float().mean(),
             "all_zero_ranking_loss": rank_loss.detach(), "sft_nll": sft_loss.detach(),
             "duplicate_fraction": loss.new_tensor(sum(duplicates) / batch),
             "completion_length": mask.sum(1).float().mean(),
             "pg_loss": -((logps.detach() * advantages * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()}
    return loss, stats


class TCAGrpoTrainer(Seq2SeqGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reference_model is None:
            raise ValueError("TCA needs a frozen SFT reference even when beta=0")
        self.gold_cache = GoldTrajectoryCache(self.grpo.gold_cache_mb)

    def compute_loss(self, model: Any, inputs: dict, return_outputs=False, num_items_in_batch=None):
        generated = self.generate_sequences(model, inputs, training=True)
        with self.accelerator.autocast():
            loss, stats = compute_tca_objective(model, self.accelerator.unwrap_model(self.reference_model), inputs,
                                              generated, self.target_index, self.grpo, self.gold_cache)
        values = self.accelerator.reduce(torch.stack([value.detach().float() for value in stats.values()]), reduction="mean").cpu().tolist()
        for name, value in zip(stats, values):
            self._rollout_metrics[name].append(value)
        return (loss, {"loss": loss}) if return_outputs else loss
