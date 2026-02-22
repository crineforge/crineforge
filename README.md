# 🚀 TrainForge

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GPU Safe](https://img.shields.io/badge/GPU-safe-orange)

Forge intelligent models from raw data.

TrainForge is a modular, offline-first LLM fine-tuning toolkit designed to automate:
- Data structuring
- Validation
- Auto hyperparameter selection
- LoRA-based fine-tuning
- GPU-safe execution
- Model saving

TrainForge works with any HuggingFace model or local model path.

---

## ✨ Features

- 🔄 Automatic data structuring (DeepSeek internal model)
- 🛡 Strict semantic preservation (no summarization)
- 📊 Token-length validation (>90% retention)
- ⚙ Auto hyperparameter configuration
- 🎯 LoRA fine-tuning support
- 🧠 GPU detection + 4bit fallback
- 🌐 Optional internet enrichment (disabled by default)
- 🔐 Offline-first architecture
- 🖥 CLI + Python API support

---

## 📦 Installation

```bash
pip install trainforge
```

Or for development:
```bash
git clone https://github.com/abhishek-dev-code/trainforge.git
cd trainforge
pip install -e .
```

---

## 🧠 Quick Start

```python
from trainforge import Trainer

trainer = Trainer()

trainer.connect_model("meta-llama/Meta-Llama-3-8B")
trainer.load_data("data.pdf")

trainer.enable_enrichment(False)  # Optional

trainer.auto_config()
trainer.train()
trainer.save("output_model")
```

That's it.

---

## 📂 Supported Data Formats

- **PDF**
- **TXT**
- **CSV**
- **JSON** (text-based)
- **Markdown**

---

## 🛠 How It Works

1. **Extract raw data**
2. **Chunk data safely**
3. **Structure data using internal DeepSeek model**
4. **Validate token preservation**
5. **Auto-detect GPU & VRAM**
6. **Configure safe hyperparameters**
7. **Fine-tune using LoRA**
8. **Save model or adapter**

---

## 🖥 GPU Handling

TrainForge automatically:
- Detects CUDA
- Detects VRAM
- Loads FP16 (>=16GB)
- Loads 4bit (>=8GB)
- Falls back to CPU safely
- Handles OOM recovery

---

## 🔐 Privacy & Security

- No external API calls required
- Fully offline after first model download
- No `eval()`
- No arbitrary shell execution
- File size limits enforced
- Deterministic seed support

---

## 🧪 Example CLI Usage

```bash
trainforge train --model meta-llama/Meta-Llama-3-8B --data data.pdf
```

---

## 📌 Roadmap

- Dataset quality scoring
- Multi-GPU support
- Dataset visualization
- Distributed training
- **Note:** TrainForge may introduce advanced commercial/SaaS tier or subscription-based features in the future, while keeping the core toolkit open-source under MIT.

---

## ⚖️ Model Usage Disclaimer

**TrainForge does not bundle or redistribute third-party model weights.**

Models (such as the internal DeepSeek structurer or any base model you choose to fine-tune) are downloaded dynamically from their official sources and are strictly subject to their respective original licenses. TrainForge code is MIT licensed, but external models are not owned or licensed by TrainForge. Users are responsible for complying with the licenses of the models they use.

Please review the [LEGAL_NOTICE.md](LEGAL_NOTICE.md) for full compliance details.

---

## 📄 License

TrainForge is licensed under the [MIT License](LICENSE). 
Copyright (c) 2025 Abhishek.
