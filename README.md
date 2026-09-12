# ConvNeXt Classification Pipeline

DIMER inference wrapper for **`timm/convnext_tiny.in12k_ft_in1k`** — ImageNet-1k image classification (1000 classes) — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `timm/convnext_tiny.in12k_ft_in1k` (ConvNeXt-Tiny, ImageNet-12k pre-train → ImageNet-1k fine-tune)
- Revision: `aa096f03029c7f0ec052013f64c819b34f8ad790`
- Upstream weight license: Apache-2.0
- Upstream task: image classification, 1000 ImageNet-1k classes, 224×224 eval input
- Repository adaptation: **none**; inference only

## Quick start

```python
from PIL import Image
from convnext_classification_pipeline import ConvNeXtClassificationPipeline, top_k_accuracy

pipe = ConvNeXtClassificationPipeline.from_pretrained()          # cuda:0 if available, else cpu
result = pipe.predict(Image.open("photo.jpg"), top_k=5)
print(result["predictions"][0]["predicted_label"], result["predictions"][0]["top_k"][0]["score"])
print(top_k_accuracy(result["predictions"], [target_index], k=1))
```

`score` is a softmax score over 1000 classes, not a calibrated probability; the reported label is the argmax.

## Weights layout

```
weights/convnext-tiny-in12k/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # timm pretrained_cfg: input size, mean/std, crop
  model.safetensors          # 114374272 bytes, git-ignored
```

`from_pretrained()` calls `verify_snapshot()` first and refuses to load if any file is missing or its SHA-256 differs from the manifest. Without a snapshot, `allow_download=True` loads from the Hub through timm's `hf-hub:timm/convnext_tiny.in12k_ft_in1k@aa096f03029c7f0ec052013f64c819b34f8ad790` form; the default is to refuse. To stage the snapshot: `hf download timm/convnext_tiny.in12k_ft_in1k --revision aa096f03029c7f0ec052013f64c819b34f8ad790 --local-dir weights/convnext-tiny-in12k`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed
```

Smoke (loads the verified snapshot and classifies one synthetic image; measured numbers are in `MODEL_CARD.md` → Runtime):

```python
from PIL import Image
from convnext_classification_pipeline import ConvNeXtClassificationPipeline

pipe = ConvNeXtClassificationPipeline.from_pretrained()
print(pipe.predict(Image.new("RGB", (256, 256), (90, 140, 200)))["predictions"][0]["predicted_label"])
```

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.0 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md`.
