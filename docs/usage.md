# TrainForge User Guide

## 1️⃣ Basic Usage

Trainer class handles everything automatically.

### Step 1: Connect model
Supports:
- HuggingFace model ID
- Local model path

### Step 2: Load data
Supported formats:
- PDF
- TXT
- CSV
- JSON
- Markdown

Data is structured internally using a strict, non-summarizing AI process.

### Step 3: Optional Enrichment
Enable internet enrichment:
```python
trainer.enable_enrichment(True)
```
(Default is False)

### Step 4: Auto Configuration
```python
trainer.auto_config()
```
Automatically selects:
- Learning rate
- Batch size
- Epochs

Based on:
- Dataset size
- GPU memory

### Step 5: Training
```python
trainer.train()
```

### Step 6: Save
```python
trainer.save("output_model")
```

---

## Strict Mode

TrainForge ensures:
- No summarization
- No semantic compression
- >90% token retention

---

## GPU Strategy

If GPU available:
- >=16GB → FP16
- >=8GB → 4bit
- else → CPU fallback

---

## Offline Mode

After first DeepSeek download:
TrainForge runs fully offline.

---

## Advanced Configuration

Manual hyperparameters:
```python
trainer.manual_config(**{
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4
})
```
