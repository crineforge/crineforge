# Crineforge Architecture

## Core Modules

### `core.py`
- Public API wrapper (`Trainer` class)
- Pipeline orchestration
- Dry-run validation limits

### `data/`
- **extractor**: Extract logic parsing formatting + 10MB bounds guard.
- **chunker**: Size limits data partitions safe for memory buffering.
- **structurer (DeepSeek internal)**: Uses local 7B Instruct variants via Singleton Lazy Loading memory cache hooks.
- **validator**: Verifies token retention >= 90% and ensures dataset bounds >= 20 parameters.
- **enrichment**: Stand-in mock hooks for future internet-bound RAG verification processes.

### `model/`
- **gpu**: GPU detection parameters mapping system hardware to VRAM usage maps.
- **connector**: HuggingFace loading bridges + LoRA adapters.

### `training/`
- **engine**: SFTTrainer runtime wrapper avoiding any remote triggers.
- **saver**: Output writing hooks.

### `hyperparams/`
- **auto**: Rules engine dynamically adjusting `batch_size`, `gradient_accumulation_steps`, and `epochs` relative to parameters size.

### `utils/`
- **logger**: Application wide std logs.
- **seed**: Seed management for RNG replication loops.
- **diffcheck**: Threshold calculation for generation limits.

---

## Internal Structuring Model

We exclusively utilize local instruction models for structuring (default: `deepseek-ai/deepseek-llm-7b-chat`).

The internal LLM is:
- Loaded locally (quantized if appropriate).
- Cached in `~/.crineforge/models`.
- Used only for structuring offline pairs.
- Not exposed to the user directly to prevent logic bloat.
- Cleanly garbage collected post-generation using Singleton reference deletion.
