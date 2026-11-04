#!/usr/bin/env python3
"""Probe the actual token-reward GRPO graph, gold reference and all-wrong fallback."""

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pretrain.calibrate_grpo_batch_size import largest_fitting_batch
from utils.grpo_data import load_corpus
from utils.grpo_trainer import TargetIndex
from utils.tca_grpo_trainer import TCASettings, GoldTrajectoryCache, compute_tca_objective


def probe_once(args, index, device, batch_size):
    model = reference = optimizer = reserve = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path, local_files_only=True).to(device)
        reference = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path, local_files_only=True).to(device)
        reference.requires_grad_(False)
        reference.eval()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            for name in ("dropout", "attention_dropout", "activation_dropout"):
                if isinstance(getattr(module, name, None), float):
                    setattr(module, name, 0.)
        settings = settings_from_args(args)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        reserve_bytes = max(int(args.reserve_gib * 1024**3), 2 * parameter_bytes if args.num_gpus > 1 else 0)
        reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16, init_scale=128)
        ordered = sorted(index.target_to_sequence, key=lambda t: len(index.target_to_sequence[t]), reverse=True)
        longest = index.target_to_sequence[ordered[0]]
        # Keep the longest generated path negative to exercise the fallback.
        positives = ordered[1:1 + settings.gold_candidates_per_query] or ordered[:1]
        content_id = next(token for token in longest if token not in (index.pad_id, index.eos_id, index.start_id))
        prompts = torch.full((batch_size, args.max_prompt_length), content_id, dtype=torch.long, device=device)
        prompts[:, -1] = index.eos_id
        inputs = {"input_ids": prompts, "attention_mask": torch.ones_like(prompts),
                  "positive_targets": [positives] * batch_size}
        config = settings.rollout_generation_config(index)

        def force_longest(batch_id, prefix):
            position = prefix.shape[0] - 1
            return [longest[position]] if position < len(longest) else [index.pad_id]

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            if args.gradient_accumulation_steps > 1:
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                generated = model.generate(input_ids=prompts, attention_mask=inputs["attention_mask"],
                                           generation_config=config, prefix_allowed_tokens_fn=force_longest)
            del generated
            generated = torch.tensor([index.start_id, *longest], device=device).repeat(batch_size * settings.num_generations, 1)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                loss, stats = compute_tca_objective(model, reference, inputs, generated, index, settings, GoldTrajectoryCache(0))
            del stats
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            scaler.step(optimizer)
            scaler.update()
            if not optimizer.state:
                raise RuntimeError("TCA probe could not allocate optimizer state")
            torch.cuda.synchronize(device)
            del loss, generated
        return True, torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
        return False, torch.cuda.max_memory_allocated(device)
    finally:
        del model, reference, optimizer, reserve


def settings_from_args(args):
    values = {name: getattr(args, name) for name in TCASettings.__dataclass_fields__ if hasattr(args, name)}
    values["do_sample"] = args.do_sample == "true"
    return TCASettings(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint_path", "corpus_file", "output_file"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--do_sample", choices=("true", "false"), default="true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--beta", type=float, default=.04)
    parser.add_argument("--gold_candidates_per_query", type=int, default=2)
    parser.add_argument("--reference_batch_size", type=int, default=16)
    parser.add_argument("--all_zero_ranking_weight", type=float, default=.1)
    parser.add_argument("--sft_anchor_weight", type=float, default=.1)
    parser.add_argument("--negatives_per_query", type=int, default=4)
    parser.add_argument("--trajectory_alignment", choices=("pre_token", "post_token"), default="pre_token")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--start_batch_size", type=int, default=4)
    parser.add_argument("--max_batch_size", type=int, default=16)
    parser.add_argument("--reserve_gib", type=float, default=4.)
    parser.add_argument("--invalid_target_policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    settings_from_args(args)
    if min(args.num_gpus, args.max_prompt_length, args.gradient_accumulation_steps) < 1 or args.reserve_gib < 0:
        parser.error("GPU count, prompt length and accumulation must be positive; reserve must be nonnegative")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        parser.error(f"Need {args.num_gpus} visible CUDA GPUs")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, local_files_only=True)
    index = TargetIndex(tokenizer, AutoConfig.from_pretrained(args.checkpoint_path, local_files_only=True),
                        load_corpus(args.corpus_file), args.max_target_length, args.invalid_target_policy)
    index.write_filter_report(Path(args.output_file).parent / "calibration_target_filtering")
    ordinal = min(range(args.num_gpus), key=lambda i: torch.cuda.mem_get_info(i)[0])
    device = torch.device("cuda", ordinal)
    torch.cuda.set_device(device)
    print(f"TCA capacity probe on cuda:{ordinal}: includes hidden states, gold reference, fallback and anchor.", flush=True)
    trials = []

    def probe(batch):
        gc.collect()
        torch.cuda.empty_cache()
        fits, peak = probe_once(args, index, device, batch)
        gc.collect()
        torch.cuda.empty_cache()
        trials.append({"queries_per_gpu": batch, "fits": fits, "peak_allocated_gib": peak / 1024**3})
        print(f"TCA batch {batch}: {'fits' if fits else 'OOM'}; peak including reserve={peak / 1024**3:.2f} GiB", flush=True)
        return fits

    chosen = largest_fitting_batch(args.start_batch_size, args.max_batch_size, probe)
    result = {"batch_size": chosen, "objective": "tca_grpo", "arguments": vars(args), "gpu": ordinal,
              "generation_rows_per_query": args.num_generations,
              "maximum_scored_rows_per_query": args.num_generations + args.gold_candidates_per_query, "trials": trials}
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
