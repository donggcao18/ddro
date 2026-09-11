#!/usr/bin/env python3
"""Train DPO, TDPO1, or TDPO2 for Vault from a Hugging Face Trainer checkpoint.

The input checkpoint must be a directory produced by ``Trainer`` or
``save_pretrained``.  In particular, a directory containing ``config.json``,
``model.safetensors`` (or ``pytorch_model.bin``), and the tokenizer files can be
passed directly to ``--checkpoint_path``.  Optimizer and trainer state files in
an SFT checkpoint are intentionally ignored; use ``--resume_from_checkpoint``
only for resuming an interrupted DPO run.

The preference data is JSON or JSONL with at least these string fields:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Extra audit fields written by ``mine_dpo_negatives.py`` are discarded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

from tdpo_trainer import PreferenceConfig, TokenDPOTrainer
from startup_diagnostics import StartupDiagnostics
from preference_cache import TokenizationCacheDataset


PREFERENCE_COLUMNS = ("prompt", "chosen", "rejected")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("value must be in the interval [0, 1)")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train encoder-decoder DPO/TDPO from a complete Hugging Face checkpoint "
            "and prompt/chosen/rejected JSONL data."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="SFT checkpoint directory containing config, weights, and tokenizer files.",
    )
    parser.add_argument("--train_file", required=True, help="Training JSON/JSONL file.")
    parser.add_argument(
        "--eval_file",
        help=(
            "Optional validation JSON/JSONL file. If omitted, a deterministic split "
            "is made from --train_file according to --validation_split."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_split", type=unit_interval, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_num_proc", type=int, default=8)
    parser.add_argument(
        "--dataset_cache_dir",
        help="Datasets cache root; use node-local disk shared by ranks on the same node.",
    )
    parser.add_argument(
        "--tokenization_writer_batch_size", type=positive_int, default=1000,
        help="Rows per Arrow write batch during TRL tokenization (default: 1000).",
    )
    parser.add_argument(
        "--ddp_timeout", type=positive_int, default=1800, metavar="SECONDS",
        help="Distributed process-group timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--startup_debug", type=int, default=0, metavar="SECONDS",
        help=(
            "Opt-in startup diagnostics: dump Python stacks every SECONDS, log rank "
            "progress, and probe distributed communication before trainer initialization. "
            "0 disables diagnostics. Use the same value on all ranks."
        ),
    )

    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=positive_float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=positive_float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=unit_interval, default=0.1)
    parser.add_argument("--beta", type=positive_float, default=0.4)
    parser.add_argument(
        "--preference_objective", choices=["dpo", "tdpo1", "tdpo2"], default="dpo"
    )
    parser.add_argument(
        "--tdpo_alpha", type=float, default=0.5,
        help="TDPO2 KL weight; ignored by DPO/TDPO1.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Override epochs, useful for a short training smoke test.",
    )
    parser.add_argument("--max_grad_norm", type=positive_float, default=0.5)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without improvement; 0 disables it.",
    )
    parser.add_argument(
        "--report_to",
        default="none",
        help="Trainer integration name (for example wandb), or 'none'.",
    )
    parser.add_argument("--run_name")

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use activation checkpointing to reduce DPO memory use (default: enabled).",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Resume DPO trainer/optimizer state. Pass no value to use the latest DPO "
            "checkpoint in output_dir, or pass an explicit DPO checkpoint directory."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate data/checkpoint loading and print tokenization examples, then exit.",
    )
    return parser.parse_args()


def validate_checkpoint(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    required = [path / "config.json"]
    weight_candidates = [
        path / "model.safetensors",
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin",
        path / "pytorch_model.bin.index.json",
    ]
    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "spiece.model",
        path / "sentencepiece.bpe.model",
    ]
    missing = [str(item.name) for item in required if not item.is_file()]
    if not any(item.is_file() for item in weight_candidates):
        missing.append("model.safetensors (or another Hugging Face model weight file)")
    if not any(item.is_file() for item in tokenizer_candidates):
        missing.append("tokenizer.json (or a SentencePiece model)")
    if missing:
        raise FileNotFoundError(
            f"{path} is not a complete Hugging Face checkpoint; missing: "
            + ", ".join(missing)
        )
    return path


def _normalize_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for column in PREFERENCE_COLUMNS:
        normalized[column] = [
            "" if value is None else str(value).strip() for value in batch[column]
        ]
    return normalized


def _valid_preference(row: dict[str, str]) -> bool:
    return bool(
        row["prompt"]
        and row["chosen"]
        and row["rejected"]
        and row["chosen"] != row["rejected"]
    )


def clean_preference_dataset(
    dataset: Dataset, split_name: str, num_proc: int
) -> Dataset:
    missing = [name for name in PREFERENCE_COLUMNS if name not in dataset.column_names]
    if missing:
        raise ValueError(
            f"The {split_name} dataset is missing required columns: {', '.join(missing)}. "
            f"Found: {', '.join(dataset.column_names)}"
        )

    original_size = len(dataset)
    dataset = dataset.map(
        _normalize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Normalizing {split_name} preferences",
    )
    dataset = dataset.filter(
        _valid_preference,
        num_proc=num_proc,
        desc=f"Validating {split_name} preferences",
    )
    dropped = original_size - len(dataset)
    if dropped:
        print(
            f"Warning: dropped {dropped:,} invalid {split_name} rows "
            "(empty fields or chosen == rejected)."
        )
    if len(dataset) == 0:
        raise ValueError(f"No valid preference rows remain in the {split_name} dataset")
    return dataset


def load_preference_datasets(args: argparse.Namespace) -> DatasetDict:
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    cache_dir = None
    if args.dataset_cache_dir:
        cache_dir = str(Path(args.dataset_cache_dir).expanduser().resolve())
        print(f"Datasets cache root: {cache_dir}", flush=True)
    datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    datasets = DatasetDict(
        {
            name: clean_preference_dataset(split, name, args.dataset_num_proc)
            for name, split in datasets.items()
        }
    )

    if "validation" not in datasets and args.validation_split:
        split = datasets["train"].train_test_split(
            test_size=args.validation_split,
            seed=args.seed,
            shuffle=True,
        )
        datasets = DatasetDict(
            {"train": split["train"], "validation": split["test"]}
        )
    return datasets


def load_tokenizer_and_config(
    checkpoint_path: Path, trust_remote_code: bool
) -> tuple[Any, Any]:
    common_kwargs = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    config = AutoConfig.from_pretrained(checkpoint_path, **common_kwargs)
    if not config.is_encoder_decoder:
        raise ValueError(
            f"Expected an encoder-decoder checkpoint, but config model_type={config.model_type!r} "
            "has is_encoder_decoder=False"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, **common_kwargs)
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "The checkpoint tokenizer has no pad token. Set and save a pad token in the SFT "
            "checkpoint before DPO training."
        )
    if config.decoder_start_token_id is None:
        raise ValueError(
            "The encoder-decoder config has no decoder_start_token_id. Save the decoder "
            "start token used during SFT in config.json before DPO training."
        )
    if (
        config.pad_token_id is not None
        and config.pad_token_id != tokenizer.pad_token_id
    ):
        raise ValueError(
            "Checkpoint pad-token mismatch: config.json uses "
            f"{config.pad_token_id}, but the tokenizer uses {tokenizer.pad_token_id}."
        )
    return tokenizer, config


def load_policy_model(checkpoint_path: Path, trust_remote_code: bool) -> Any:
    return AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )


def validate_model_tokenizer(model: Any, tokenizer: Any) -> None:
    embedding_count = model.get_input_embeddings().num_embeddings
    if embedding_count < len(tokenizer):
        raise ValueError(
            "Checkpoint/tokenizer vocabulary mismatch: model has "
            f"{embedding_count:,} embeddings but tokenizer has {len(tokenizer):,} tokens. "
            "Use the tokenizer saved in the same SFT checkpoint; do not resize the DPO model."
        )


def print_preflight(
    datasets: DatasetDict,
    tokenizer: Any,
    config: Any,
    max_prompt_length: int,
    max_target_length: int,
) -> None:
    summary = {name: len(dataset) for name, dataset in datasets.items()}
    print("Dataset sizes:", json.dumps(summary, sort_keys=True))
    print(
        "Checkpoint:",
        json.dumps(
            {
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "decoder_start_token_id": config.decoder_start_token_id,
                "pad_token_id": config.pad_token_id,
                "eos_token_id": config.eos_token_id,
                "tokenizer_size": len(tokenizer),
            },
            sort_keys=True,
        ),
    )

    examples = datasets["train"].select(range(min(3, len(datasets["train"]))))
    for index, row in enumerate(examples):
        prompt_ids = tokenizer(
            row["prompt"], truncation=True, max_length=max_prompt_length
        )["input_ids"]
        chosen_ids = tokenizer(
            row["chosen"], truncation=True, max_length=max_target_length
        )["input_ids"]
        rejected_ids = tokenizer(
            row["rejected"], truncation=True, max_length=max_target_length
        )["input_ids"]
        print(
            f"Example {index}: prompt_tokens={len(prompt_ids)}, "
            f"chosen={row['chosen']!r}->{chosen_ids}, "
            f"rejected={row['rejected']!r}->{rejected_ids}"
        )


def build_training_args(args: argparse.Namespace, has_eval: bool) -> DPOConfig:
    report_to = [] if args.report_to.lower() == "none" else [args.report_to]
    strategy = "steps" if has_eval else "no"
    return PreferenceConfig(
        preference_objective=args.preference_objective,
        tdpo_alpha=args.tdpo_alpha,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_timeout=args.ddp_timeout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_prompt_length + args.max_target_length,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
        truncation_mode="keep_end",
        logging_steps=args.logging_steps,
        evaluation_strategy=strategy,
        eval_steps=args.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.dataset_num_proc,
        remove_unused_columns=False,
        report_to=report_to,
    )


def main() -> None:
    args = parse_args()
    with StartupDiagnostics(args.startup_debug) as debug:
        run_training(args, debug)


def run_training(args: argparse.Namespace, debug: StartupDiagnostics) -> None:
    if args.max_prompt_length <= 0 or args.max_target_length <= 0:
        raise ValueError("Token length limits must be greater than zero")
    if args.dataset_num_proc <= 0:
        raise ValueError("--dataset_num_proc must be greater than zero")
    if args.save_steps <= 0 or args.eval_steps <= 0 or args.logging_steps <= 0:
        raise ValueError("Logging, evaluation, and save step counts must be greater than zero")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative")

    debug.mark("Validating SFT checkpoint")
    checkpoint_path = validate_checkpoint(args.checkpoint_path)
    set_seed(args.seed)
    debug.mark("Loading and cleaning preference data BEGIN")
    datasets = load_preference_datasets(args)
    debug.mark("Loading and cleaning preference data DONE")
    has_eval = "validation" in datasets
    debug.mark("Training arguments / distributed initialization BEGIN")
    training_args = build_training_args(args, has_eval)
    if args.startup_debug:
        debug.probe(training_args.device)
    debug.mark("Training arguments / distributed initialization DONE")
    print(
        f"Preference objective: {args.preference_objective}, "
        f"beta={args.beta}, alpha={args.tdpo_alpha}"
    )
    debug.mark("Loading tokenizer and config BEGIN")
    tokenizer, config = load_tokenizer_and_config(
        checkpoint_path, args.trust_remote_code
    )
    print_preflight(
        datasets,
        tokenizer,
        config,
        args.max_prompt_length,
        args.max_target_length,
    )

    debug.mark("Loading policy model BEGIN")
    model = load_policy_model(checkpoint_path, args.trust_remote_code)
    validate_model_tokenizer(model, tokenizer)
    debug.mark("Loading policy model DONE")
    if args.dry_run:
        print(
            "Dry run successful: checkpoint, tokenizer, and preference data are compatible."
        )
        return

    # DPO compares the trainable policy against a frozen copy of the exact SFT
    # checkpoint. Trainer checkpoint files such as optimizer.pt are not loaded here.
    debug.mark("Loading reference model BEGIN")
    reference_model = load_policy_model(checkpoint_path, args.trust_remote_code)
    reference_model.requires_grad_(False)
    reference_model.eval()
    debug.mark("Loading reference model DONE")

    if args.gradient_checkpointing:
        model.config.use_cache = False

    if has_eval and args.save_steps % args.eval_steps:
        raise ValueError(
            "When validation is enabled, --save_steps must be a multiple of "
            "--eval_steps so load_best_model_at_end can select a matching checkpoint."
        )
    callbacks = []
    if has_eval and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer_class = DPOTrainer if args.preference_objective == "dpo" else TokenDPOTrainer
    datasets = DatasetDict({
        name: TokenizationCacheDataset.from_dataset(split, args.tokenization_writer_batch_size)
        for name, split in datasets.items()
    })
    debug.mark("Trainer initialization BEGIN (includes TRL tokenization and barriers)")
    trainer = trainer_class(
        model=model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        tokenizer=tokenizer,
        is_encoder_decoder=True,
        callbacks=callbacks,
    )
    debug.mark("Trainer initialization DONE; entering train/resume")
    debug.stop()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Keep checkpoints resumable in output_dir and place the portable inference
    # artifact in final/.  This has the same config/weights/tokenizer layout as SFT.
    final_dir = Path(args.output_dir) / "final"
    trainer.model.config.use_cache = True
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    print(f"Saved final Hugging Face DPO model to {final_dir}")


if __name__ == "__main__":
    main()
