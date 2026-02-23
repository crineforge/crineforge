# CrineForge

Forge intelligent text-trained models from raw documents.

---

## What is CrineForge?

CrineForge is a lightweight, offline-first LLM fine-tuning toolkit
that automatically structures raw text data using a DeepSeek-based
structurer and fine-tunes HuggingFace models using LoRA.

It is designed to be safe, modular, and GPU-aware.

---

## Who is it for?

- ML engineers
- AI developers
- Local model fine-tuning users
- Developers who want structured training without heavy setup

---

## Key Features

- DeepSeek-based structured data preparation
- LoRA fine-tuning support
- Automatic 4-bit fallback for low VRAM GPUs
- Gradient checkpointing
- VRAM usage logging
- CLI + Python API
- Offline-first design

---

## Quickstart

```bash
pip install crineforge
```

```python
from crineforge import Trainer

trainer = Trainer()
trainer.connect_model("sshleifer/tiny-gpt2")
trainer.load_data("data.txt")
trainer.auto_config()
trainer.train()
trainer.save("output_model")
```

---

## Performance & Optimization

- Lazy-loaded structurer exclusively deployed at generation time.
- Structurer unloads natively before fine-tuning prevents VRAM spikes.
- Explicit `gradient_checkpointing` automatically supported.
- Automatic 4-bit quantization fallback detecting VRAM `< 16GB`.
- Comprehensive VRAM Logging tracking Allocated & Reserved metrics.
- Default `max_seq_length = 512` enforcing scalable SFT executions.

---

### 🔥 Important Disclaimer

```markdown
Note:
CrineForge does not redistribute DeepSeek weights.
Models are downloaded from their official sources and are subject to their respective licenses.
```

---

## 📄 License

Crineforge is licensed under the [MIT License](LICENSE). 
Copyright (c) 2025 Abhishek.
