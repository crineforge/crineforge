<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src=".assets/logo.png" width="200" alt="Crineforge Logo" style="display: block; margin: 0 auto;" />
  <h1 style="font-size: 3rem; margin-top: 15px;">Crineforge</h1>
  <p style="font-size: 1.2rem; margin-top: -10px;"><b>Forge intelligent parameter-efficient models from raw documents with enterprise-grade reliability.</b></p>
</div>

<hr>

<div align="center" style="line-height: 1;">
  <a href="https://pypi.org/project/crineforge/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/crineforge?color=536af5&logoColor=white"/></a>
  <a href="https://github.com/crineforge/crineforge"><img alt="Python Version" src="https://img.shields.io/pypi/pyversions/crineforge?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/crineforge/crineforge/blob/main/LICENSE"><img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53"/></a>
  <br>
  <a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Integration-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-EE4C2C?logo=pytorch&logoColor=white&color=EE4C2C"/></a>
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Crineforge Ecosystem Strategy](#2-crineforge-ecosystem-strategy)
3. [Core Capabilities](#3-core-capabilities)
4. [Hardware Requirements](#4-hardware-requirements)
5. [How to Run Locally](#5-how-to-run-locally)
6. [Detailed Documentation](#6-detailed-documentation)
7. [License](#7-license)
8. [Contact](#8-contact)

## 1. Introduction

We present **Crineforge**, an offline-first execution framework engineered for end-to-end Large Language Model instruction tuning. 

Crineforge was built for organizations requiring uncompromising privacy when handling internal data lakes. It automatically handles the heavy lifting of raw document traversal, robust context bounding, dynamic VRAM mapping, and formatting bounds—without exposing complex boilerplate to the end user.

By natively integrating **local instruction-following AI** proxy servers (`Qwen2.5-1.5B-Instruct` or `DeepSeek-7B-Chat`), Crineforge securely transforms untidy company documents (PDF, CSV, TXT) into cleanly mapped LoRA fine-tuning matrices, acting entirely offline within your local environment.

## 2. Crineforge Ecosystem Strategy

This repository hosts the **Crineforge Community Edition**, which contains our core data pipeline and offline structuring algorithms. For production workflows, we offer expanding layers of scale.

---

**Architecture: Offline Structuring & Verification**
- We pioneer an offline Validation Gateway powered by open-source Instruct models. By spinning up local model instances natively inside the framework, we extract functional `<instruction, response>` pairs securely without transmitting proprietary data to external APIs.
- The framework manages structural fidelity internally, safely bypassing anomalies to maintain highly reliable synthetic structures essential for downstream convergence.

---

**Fine-Tuning: Hardware Independence**
- We integrate Auto-Config profiling tools to dynamically negotiate precision and training epochs against available target VRAM constraints. 

---

### Exploring Crineforge Enterprise Solutions
For enterprise teams requiring hosted integrations, heavy offline RAG-validation routing, API distribution, and dedicated support, our **Pro & Enterprise Modules** expand on these core limits heavily. Ensure you follow our repository for future advanced deployment options.

## 3. Core Capabilities

<div align="center">

| Feature | Community Edition | Enterprise / Pro (Coming Soon) |
| :------------ | :------------: | :------------: |
| **Document Traversal Engine** | Standard File Execution (PDF/TXT) | Distributed Document Lakes |
| **Data Fidelity Check** | Fault-Tolerant Heuristic Filtering | Agentic Validation & Auto-healing |
| **Precision Scaling** | Automatic Consumer 4-bit Fallbacks | Custom Distributed Tensor Routing |
| **Local Structurer Model** | `Qwen2.5 1.5B` & `DeepSeek 7B` | Swarm-Based Multi-Agent Structuring |

</div>

<br>

## 4. Hardware Requirements

### Target Device Guidelines

<div align="center">

| Physical GPU Setup | Minimum VRAM Needed | Target Precision | Performance Bracket |
| :------------ | :------------ | :------------: | :------------: |
| **Consumer Terminal/Laptop** | 8 GB | Auto config (default 4-bit) | Developer Sandbox |
| **Standard Workstation** | 16 GB | BF16/FP16 native arrays | High Throughput |
| **A100/H100 Node** | 24+ GB | FP16/BF16 + DeepSeek Proxy | Enterprise Grade |

</div>

## 5. How to Run Locally

Crineforge runs seamlessly across local desktop clusters and isolated offline pods via Pip. 

### 5.1 Installation

#### System Requirements

> [!NOTE] 
> Python `3.10` or higher is natively required, backed by PyTorch 2.0+ compatible with local CUDA environments.

Install the framework globally:
```shell
pip install crineforge
```

### 5.2 Launching the Pipeline

Instantiate your pipeline safely:

```python
import os
from crineforge import Trainer

# Initialize the community orchestration engine
trainer = Trainer()
trainer.connect_model("sshleifer/tiny-gpt2")

# Route untidy system files into the Extractor Engine
trainer.load_data("internal_docs.pdf")

# Crineforge executes offline mapping and begins local training
trainer.auto_config()
trainer.train()

# Emit target LoRA ready for inference
trainer.save("deploy_lora_v1")
```

### 5.3 Pro Proxy Mode (Heavyweight Servers)

For powerful distributed workstations offering generous VRAM boundaries (>24GB), you may directly delegate data transformations to the heavier `deepseek-llm-7b-chat` proxy:

```python
# Boot the pipeline specifying the heavier DeepSeek offline tutor
trainer = Trainer(structurer_model="deepseek-ai/deepseek-llm-7b-chat")
```

## 6. Detailed Documentation

For a technical overview into the framework mechanics and architecture protocols, please review our comprehensive guides:

- [📖 Workflow & Usage Directives](docs/usage.md): Manual overrides, hardware instructions, and internal execution paths.
- [⚙️ Technical Blueprint](docs/architecture.md): Visual pathways of the internal Pipeline Gateways.

## 7. License
Crineforge is licensed via the [MIT License](LICENSE), built tightly around community-driven domains. The internal structurer models pulled (such as Qwen and DeepSeek) operate under their explicit distinct Model Licenses, securing safety from licensing bottlenecks.

Crineforge explicitly guarantees complete safety for corporate internal exploitation.

## 8. Contact
For commercial inquiries, enterprise setup evaluations, or specific issue tracking related to dataset bounds limits, please reference our official issue boards.
