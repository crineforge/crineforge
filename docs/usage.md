# 📖 Crineforge User Guide & Operations

Welcome to the definitive operating manual for **Crineforge**, the execution engine bridging unstructured data to high-capacity LoRA integrations offline.

## 1. Quick Launch Operations

Our `Trainer` engine handles complex state and orchestration natively. A highly scaled deployment can be executed fully locally.

### Context Initialization
```python
from crineforge import Trainer

# Optional Configuration: Designate custom Community Architectures
# 'deepseek-ai/deepseek-llm-7b-chat' represents our high-scale, >24GB VRAM structural target.
trainer = Trainer(structurer_model="Qwen/Qwen2.5-1.5B-Instruct") 
```

### Extractor Engine Routing
Direct your unstructured formats natively:
- `.pdf` (Internal visually-bounded traversal)
- `.txt` / `.md`
- `.csv` / `.json`

```python
trainer.load_data("system_logs.pdf")
```
_All extraction pipelines pass through highly bounded internal algorithms that protect data boundaries and enforce factual fidelity rules, preventing LLM summary hallucination loops._

---

## 2. Adaptive Hardware Balancing

Instead of manually calibrating gradient thresholds, delegate resource negotiation to Crineforge:

```python
# Crineforge inspects document token arrays against available VRAM nodes.
trainer.auto_config()
```

### Manual Engine Bypass
For users requiring explicit training constraints:
```python
trainer.manual_config(**{
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4, 
    "gradient_accumulation_steps": 2
})
```

---

## 3. Offline Execution Protocols

In domains requiring massive factual privacy, Crineforge defaults strictly to **Air-Gapped Operational Targets**.
1. **Model Persistence**: `Qwen` and `DeepSeek` proxies are pulled once, verified, and mapped to internal cache memory paths.
2. **True Null Exfiltration**: Following validation against the local cache, extraction parsing execution triggers completely locally. Absolutely zero context arrays are sent to hosted endpoints.

### Pre-Flight Verification
Execute offline validation rules natively without activating tuning arrays:
```python
trainer.dry_run() 
```

---

## 4. Advanced System Modules & Enrichment

Crineforge includes heuristic validations capable of dynamic scaling. 

```python
# Engages advanced routing hooks required for RAG validation sweeps.
trainer.enable_enrichment(True)
```
> [!NOTE]
> Enrichment Hooks and heuristic loops serve as the foundational bedrock for our forthcoming **Pro Validation Engines**, targeting massive external data reconciliations.

---

## 5. Artifact Commitment

```python
trainer.train() # Invokes chunking buffers, offline formatting, tensor mapping, and active tuning.
```

```python
# Extract the production-ready LoRA Adapter block
trainer.save("deploy_lora_production_v1")
```
