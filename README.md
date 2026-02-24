# CrineForge

Forge intelligent text-trained models from raw documents with enterprise-grade reliability.

---

## 🚀 What is CrineForge?

CrineForge is a lightweight, offline-first LLM fine-tuning toolkit designed to take you from **raw documents to a fine-tuned LoRA model** in minutes. It automatically structures raw text data using a powerful local structurer and fine-tunes HuggingFace models seamlessly.

It is designed to be safe, modular, and GPU-aware, providing exceptional performance out of the box.

---

## 🎯 Who is it for?

- **ML Engineers & AI Developers** needing rapid, reliable fine-tuning pipelines.
- **Local Sandbox Users** testing models securely on private data.
- **Enterprise Operations** wanting structured training without heavy, complex configuration frameworks.

---

## ✨ Key Features

- **Blazing Fast Structuring:** Powered by `Qwen/Qwen2.5-1.5B-Instruct` out-of-the-box for minimal VRAM footprint and high-speed JSON generation.
- **Pro Mode Structuring:** Optional DeepSeek 7B fallback for rigorous, enterprise-scale formatting.
- **LoRA Fine-Tuning Support:** Native integration with `trl` and `peft`.
- **Automatic 4-bit Fallback:** Zero-configuration fallback quantization for low VRAM GPUs.
- **Gated Model Support:** First-class support for `HF_TOKEN` authenticated models (e.g., Llama-3).
- **VRAM Logging:** Detailed tracking of Allocated & Reserved memory metrics.

---

## ⚙️ System Requirements & Limitations

### VRAM Requirements (Estimated)
| GPU VRAM | Mode Availability |
| :--- | :--- |
| **8 GB** | 4-bit LoRA (Default fallback) |
| **16 GB** | FP16/BF16 LoRA |
| **24+ GB** | Pro Mode Structurer (DeepSeek 7B) + FP16 Training |

*Note: VRAM usage varies depending on context length and batch size.*

### Limitations
- The default `max_seq_length` is conservatively set to `512` to prevent OOM errors on standard hardware.
- Structurer models require an initial download which may take time depending on your network.

---

## ⚡ Quickstart

```bash
pip install crineforge
```

```python
import os
from crineforge import Trainer

# Optional: Enable authenticated access to gated models
# os.environ["HF_TOKEN"] = "your_huggingface_token"

trainer = Trainer()
trainer.connect_model("sshleifer/tiny-gpt2")
trainer.load_data("data.txt")
trainer.auto_config()
trainer.train()
trainer.save("output_model")
```

### 🧠 Pro Mode (Heavyweight Structurer)
For power users with abundant VRAM, you can enable the DeepSeek 7B structurer:
```python
trainer = Trainer(structurer_model="deepseek-ai/deepseek-llm-7b-chat")
```

---

## 📊 Performance & Optimization

- **Efficient Structurer:** The default lightweight structurer (`Qwen 1.5B`) is utilized for performance, avoiding the heavy VRAM constraints of larger models.
- **Lazy-Loaded:** The structurer is exclusively deployed at generation time.
- **VRAM Clearance:** The structurer unloads natively *before* fine-tuning begins to prevent VRAM spikes.
- **Checkpointing:** Explicit `gradient_checkpointing` automatically supported.

---

### 🔥 Important Disclaimer

```markdown
Note:
CrineForge does not redistribute model weights.
Models are downloaded from their official sources and are subject to their respective licenses.
```

---

## 📄 License

Crineforge is licensed under the [MIT License](LICENSE). 
Copyright (c) 2025 Abhishek.
