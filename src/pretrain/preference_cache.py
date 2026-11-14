"""Cache-layout adapter for TRL 0.11.4's hard-coded tokenization writer size."""

from copy import deepcopy
import os
import time
from types import SimpleNamespace

from datasets import Dataset
from datasets.fingerprint import Hasher


class TokenizationCacheDataset(Dataset):
    @classmethod
    def from_dataset(cls, dataset, writer_batch_size):
        if writer_batch_size <= 0:
            raise ValueError("Tokenization writer batch size must be positive")
        # Preserve selection indices and row order, including validation splits.
        # Namespace the fingerprint so old, fragmented caches are not reused.
        fingerprint = Hasher.hash(("ddro-token-cache-v1", dataset._fingerprint, writer_batch_size))
        wrapped = cls(
            dataset.data,
            info=deepcopy(dataset.info),
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=fingerprint,
        )
        wrapped.set_format(**dataset.format)
        wrapped.tokenization_writer_batch_size = writer_batch_size
        return wrapped

    def map(self, function=None, *args, **kwargs):
        is_trl_tokenization = (
            getattr(function, "__module__", None) == "trl.trainer.dpo_trainer"
            and getattr(function, "__name__", None) == "_tokenize"
        )
        if not is_trl_tokenization:
            return super().map(function, *args, **kwargs)
        if args:
            raise ValueError("Expected TRL tokenization map options as keyword arguments")
        kwargs["writer_batch_size"] = self.tokenization_writer_batch_size
        options = kwargs.get("fn_kwargs", {})
        if options.get("model") is not None and options.get("processor") is None:
            # TRL 0.11.4 uses the model only to select its encoder-decoder path.
            # Keep model weights and rank-specific training state out of the cache hash.
            config = options["args"]
            kwargs["fn_kwargs"] = {
                **options,
                "model": True,
                "args": SimpleNamespace(
                    max_prompt_length=config.max_prompt_length,
                    max_completion_length=config.max_completion_length,
                ),
            }
        prefix = f"[token-cache rank={os.environ.get('RANK', '0')}]"
        description = kwargs.get("desc") or "Tokenization"
        print(f"{prefix} {description}; writer_batch_size={self.tokenization_writer_batch_size}; mapping BEGIN", flush=True)
        started = time.monotonic()
        result = super().map(function, **kwargs)
        print(
            f"{prefix} {description}; mapping/cache reopen DONE in {time.monotonic() - started:.1f}s; "
            f"cache_files={result.cache_files}",
            flush=True,
        )
        return result
