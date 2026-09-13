#!/usr/bin/env python3
"""Train DDRO/DPO for Vault from a Hugging Face Trainer checkpoint.

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
    TrainerCallback,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

try:
    from .iterative_dpo_utils import atomic_json, checkpoint_hash, file_hash, read_json, verify_round_inputs
except ImportError:  # Direct script entry point.
    from iterative_dpo_utils import atomic_json, checkpoint_hash, file_hash, read_json, verify_round_inputs


PREFERENCE_COLUMNS = ("prompt", "chosen", "rejected")


def positive_float(value: str) -> float:
    parsed = float(value)
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
            "Train encoder-decoder DPO from a complete Hugging Face checkpoint "
            "and prompt/chosen/rejected JSONL data."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="SFT checkpoint directory containing config, weights, and tokenizer files.",
    )
    parser.add_argument("--train_file", required=True, help="Training JSON/JSONL file.")
    parser.add_argument("--reference_checkpoint_path", help="Frozen reference; defaults to checkpoint_path")
    parser.add_argument("--max_steps", type=int, default=-1, help="Optimizer steps; overrides epochs when positive")
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export_latest", action="store_true", help="Export latest/; requires --no-load_best_model_at_end")
    parser.add_argument("--strict_targets", action="store_true", help="Reject truncated/colliding decoder targets")
    parser.add_argument("--round_manifest", help="Immutable controller manifest required for iterative resume")
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

    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--target_length_policy", choices=["error", "skip", "truncate"], default="error")
    parser.add_argument("--target_collision_policy", choices=["error", "skip", "allow"], default="error")
    parser.add_argument("--target_token_policy", choices=["error", "allow"], default="error")
    parser.add_argument("--num_train_epochs", type=positive_float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=positive_float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=unit_interval, default=0.1)
    parser.add_argument("--beta", type=positive_float, default=0.4)
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

    datasets = load_dataset("json", data_files=data_files)
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
    return DPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        load_best_model_at_end=has_eval and args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.dataset_num_proc,
        remove_unused_columns=False,
        report_to=report_to,
        precompute_ref_log_probs=False,
        sync_ref_model=False,
    )


def validate_target_sequences(datasets: DatasetDict, tokenizer: Any, limit: int,
                              length_policy: str = "error", collision_policy: str = "error",
                              token_policy: str = "error") -> None:
    targets = set()
    for dataset in datasets.values():
        targets.update(dataset["chosen"])
        targets.update(dataset["rejected"])
    owners = {}
    for target in sorted(targets):
        options = {"truncation": length_policy == "truncate", "add_special_tokens": True}
        if length_policy == "truncate":
            options["max_length"] = limit
        sequence = tuple(tokenizer(target, **options)["input_ids"])
        if not sequence or len(sequence) > limit:
            raise ValueError(f"Decoder target must fit without truncation ({len(sequence)} > {limit}): {target!r}")
        if tokenizer.eos_token_id is not None and sequence[-1] != tokenizer.eos_token_id:
            raise ValueError(f"Target does not end in EOS: {target!r}")
        if token_policy != "allow" and (tokenizer.pad_token_id in sequence or (
            tokenizer.unk_token_id is not None and tokenizer.unk_token_id in sequence
        )):
            raise ValueError(f"Target contains padding/unknown tokens: {target!r}")
        if collision_policy != "allow" and sequence in owners and owners[sequence] != target:
            raise ValueError(f"Token-identical targets: {owners[sequence]!r}, {target!r}")
        owners[sequence] = target


class RoundCheckpointCallback(TrainerCallback):
    def __init__(self, identity: str):
        self.identity = identity

    def on_save(self, args, state, control, **kwargs):
        # on_save runs after the trainer has written optimizer/scheduler/RNG.
        from accelerate.utils import wait_for_everyone
        wait_for_everyone()
        if state.is_world_process_zero:
            atomic_json(Path(args.output_dir) / f"checkpoint-{state.global_step}" / "round_checkpoint.json",
                        {"identity": self.identity, "global_step": state.global_step,
                         "world_size": args.world_size})


def disable_t5_attention_dropout(model: Any) -> None:
    # TRL disables nn.Dropout modules, but T5 attention also uses functional
    # dropout with a float probability. Disable that path for matching policy
    # and reference likelihoods, including while the policy is in train mode.
    from transformers.models.t5.modeling_t5 import T5Attention
    for module in model.modules():
        if isinstance(module, T5Attention):
            module.dropout = 0.0


def train_round(args: argparse.Namespace) -> None:
    if args.max_prompt_length <= 0 or args.max_target_length <= 0:
        raise ValueError("Token length limits must be greater than zero")
    if args.dataset_num_proc <= 0:
        raise ValueError("--dataset_num_proc must be greater than zero")
    if args.save_steps <= 0 or args.eval_steps <= 0 or args.logging_steps <= 0:
        raise ValueError("Logging, evaluation, and save step counts must be greater than zero")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 or positive")
    if args.export_latest and args.load_best_model_at_end:
        raise ValueError("--export_latest requires --no-load_best_model_at_end")
    if args.early_stopping_patience and not args.load_best_model_at_end:
        raise ValueError("Early stopping requires best-model loading; use round-level retrieval selection instead")

    identity = None
    if args.round_manifest:
        manifest = read_json(args.round_manifest)
        verify_round_inputs(manifest)
        identity = file_hash(args.round_manifest)
        if args.validation_split or args.eval_file or not args.export_latest or not args.strict_targets:
            raise ValueError("Iterative rounds require fixed training pairs, strict targets, and latest export")
        if Path(args.checkpoint_path).resolve() != Path(manifest["policy_checkpoint"]).resolve():
            raise ValueError("Policy checkpoint differs from round manifest")
        if Path(args.train_file).resolve() != Path(manifest["pairs_file"]).resolve():
            raise ValueError("Training pairs differ from round manifest")
        for key, value in manifest["training"].items():
            if getattr(args, key) != value:
                raise ValueError(f"Training option differs from round manifest: {key}")
        precision = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"
        if precision != manifest["precision"]:
            raise ValueError("Training precision differs from round manifest")
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is True:
                raise ValueError("Iterative resume requires an explicit completed checkpoint")
            marker = Path(args.resume_from_checkpoint) / "round_checkpoint.json"
            if not marker.is_file() or read_json(marker)["identity"] != identity:
                raise ValueError("Resume checkpoint has a different round identity")

    checkpoint_path = validate_checkpoint(args.checkpoint_path)
    reference_path = validate_checkpoint(args.reference_checkpoint_path or args.checkpoint_path)
    if identity and (reference_path.resolve() != Path(manifest["reference_checkpoint"]).resolve()
                     or checkpoint_hash(reference_path) != manifest["reference_fingerprint"]):
        raise ValueError("Reference differs from the frozen round reference")
    set_seed(args.seed)
    datasets = load_preference_datasets(args)
    tokenizer, config = load_tokenizer_and_config(
        checkpoint_path, args.trust_remote_code
    )
    if args.strict_targets:
        validate_target_sequences(datasets, tokenizer, args.max_target_length,
                                  args.target_length_policy, args.target_collision_policy,
                                  args.target_token_policy)
    print_preflight(
        datasets,
        tokenizer,
        config,
        args.max_prompt_length,
        args.max_target_length,
    )

    model = load_policy_model(checkpoint_path, args.trust_remote_code)
    validate_model_tokenizer(model, tokenizer)
    if args.dry_run:
        print(
            "Dry run successful: checkpoint, tokenizer, and preference data are compatible."
        )
        return

    # Load the round's frozen reference independently from the latest policy.
    # Trainer checkpoint files such as optimizer.pt are not loaded here.
    reference_tokenizer, reference_config = load_tokenizer_and_config(reference_path, args.trust_remote_code)
    if tokenizer.get_vocab() != reference_tokenizer.get_vocab() or any(
        getattr(config, key) != getattr(reference_config, key)
        for key in ("decoder_start_token_id", "eos_token_id", "pad_token_id", "vocab_size")
    ):
        raise ValueError("Policy/reference tokenizer or configuration mismatch")
    reference_model = load_policy_model(reference_path, args.trust_remote_code)
    validate_model_tokenizer(reference_model, tokenizer)
    reference_model.requires_grad_(False)
    reference_model.eval()
    if args.round_manifest:
        disable_t5_attention_dropout(model)
        disable_t5_attention_dropout(reference_model)

    if args.gradient_checkpointing:
        model.config.use_cache = False

    has_eval = "validation" in datasets
    if has_eval and args.load_best_model_at_end and args.save_steps % args.eval_steps:
        raise ValueError(
            "When validation is enabled, --save_steps must be a multiple of "
            "--eval_steps so load_best_model_at_end can select a matching checkpoint."
        )
    training_args = build_training_args(args, has_eval)
    callbacks = []
    if identity:
        callbacks.append(RoundCheckpointCallback(identity))
    if has_eval and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        tokenizer=tokenizer,
        is_encoder_decoder=True,
        callbacks=callbacks,
    )
    if identity and trainer.accelerator.num_processes != manifest["num_gpus"]:
        raise ValueError("Training process count differs from round manifest")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Keep checkpoints resumable in output_dir and place the portable inference
    # artifact in final/.  This has the same config/weights/tokenizer layout as SFT.
    final_dir = Path(args.output_dir) / ("latest" if args.export_latest else "final")
    trainer.model.config.use_cache = True
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        atomic_json(Path(args.output_dir) / "training_metrics.json", result.metrics)
        if identity:
            atomic_json(final_dir / "round_complete.json", {
                "identity": identity, "global_step": trainer.state.global_step,
                "checkpoint_fingerprint": checkpoint_hash(final_dir),
            })
    trainer.accelerator.wait_for_everyone()
    print(f"Saved final Hugging Face DPO model to {final_dir}")


def main() -> None:
    train_round(parse_args())


if __name__ == "__main__":
    main()
