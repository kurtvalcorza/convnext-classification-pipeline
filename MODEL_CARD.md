---
license: apache-2.0
model_card_spec: "1.0"
pipeline_tag: image-classification
base_model: timm/convnext_tiny.in12k_ft_in1k
---

# ConvNeXt-Tiny in12k_ft_in1k (DIMER package v0.1.0)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fconvnext__tiny.in12k__ft__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k)
[![GitHub](https://img.shields.io/badge/GitHub-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![arXiv](https://img.shields.io/badge/arXiv-2201.03545-b31b1b.svg)](https://arxiv.org/abs/2201.03545)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

###### Description

`timm/convnext_tiny.in12k_ft_in1k` is the ConvNeXt-Tiny convolutional classifier trained by Ross Wightman in `timm`: pre-trained on ImageNet-12k (an 11821-class subset of ImageNet-22k) and then fine-tuned on ImageNet-1k (upstream README), pinned here to revision `aa096f03029c7f0ec052013f64c819b34f8ad790`. ConvNeXt (Liu et al., arXiv:2201.03545) is a pure convolutional network modernised with Transformer-era design choices: a 4×4 stride-4 patchify stem, four stages whose feature maps are 96×56×56, 192×28×28, 384×14×14 and 768×7×7 at 224 px (upstream README feature-map example), blocks built from a 7×7 depthwise convolution, LayerNorm and an inverted-bottleneck MLP with GELU, then global average pooling, a final LayerNorm and a 1000-way linear head (`head.fc` in the snapshot `config.json`). Upstream reports 28.6 M parameters and 4.5 GMACs at 224 px. Inference maps a normalised 3×224×224 tensor to 1000 logits in one forward pass; nothing is adapted, fine-tuned or conditioned in this repository. What this repository adds is packaging: the `ConvNeXtClassificationPipeline` class in `src/convnext_classification_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`), input validation, a fixed output contract and a `top_k_accuracy` helper.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is single-label image classification: input one PIL image or a batch of up to `MAX_BATCH = 64` images; output, per image, the `top_k` (default 5) ImageNet-1k classes with their softmax scores plus the argmax label. Envisioned applications are general object and scene tagging on photographs that resemble ImageNet — consumer photos, product catalogues, wildlife camera frames, stock-image indexing — where a modern CNN with ImageNet-12k pre-training is wanted at roughly the compute of a ResNet-50 (4.5 vs 4.1 GMACs per the two upstream cards). In a larger system the pipeline is an inference component, not a decision engine; the 768-d pooled features are not exposed by this package (the DINOv2 sibling covers feature extraction).

###### Primary Intended Users

The intended users are machine-learning engineers, data scientists and application developers integrating a classifier into research prototypes, internal enterprise tooling, or the DIMER model workbench. The pipeline assumes its users understand that the label space is fixed to the 1000 ImageNet-1k classes, that a softmax score is not a calibrated probability, that images far from the ImageNet distribution produce confident-looking nonsense, and that any deployment on their own data needs a labelled evaluation set. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not object detection, segmentation, multi-label tagging, OCR, or open-vocabulary classification; anything outside the 1000 ImageNet-1k classes cannot be named. The ImageNet-12k pre-training vocabulary is not exposed — the fine-tuned head answers only the 1k classes. Feature extraction is not exposed here — use `dinov2-feature-extraction-pipeline`.
2. **Input boundary:** only PIL images are accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; batches above 64 are rejected; every image is resized to 235 px and centre-cropped to 224×224 (`crop_pct = 0.95` from the snapshot `config.json`), so fine detail in large images is lost. The upstream 288-px `test_input_size` is not used by this pipeline. Non-RGB modes are converted to RGB; depth, multispectral and video inputs are unsupported.
3. **Decision boundary:** not for autonomous or high-impact decisions — content moderation takedowns, safety interlocks, medical or forensic triage — without a human reviewing the prediction and a locally measured error rate.

#### Factors

###### Groups

The pipeline is not human-centric: it is an object-centric classifier whose label space contains no person-identity, age, gender or skin-type categories. ImageNet-1k and the larger ImageNet-12k pre-training set nevertheless contain many images of people, and ImageNet has documented label problems (ambiguous, offensive and mislabelled categories in the original hierarchy). Neither the upstream `timm` card nor this repository reports any group-level performance breakdown, and neither training set is group-audited. The fairness audit therefore transfers to the operator: before deployment, measure `top_k_accuracy` on a labelled sample of your own data stratified by the groups that matter to your application, and treat any material gap as a blocker.

###### Instrumentation

ImageNet images were collected from web image searches (Deng et al., 2009) and are consumer camera photographs of varied, undocumented provenance — many makes of camera, lens and post-processing, mostly JPEG-encoded. The pipeline consumes decoded pixel arrays, so the instrument sits behind PIL: resolution, JPEG compression level, colour profile, white balance and sensor noise all reach the model as changed pixel statistics after the 224-px resize. The pipeline does not detect drift, blur, over-exposure or a change of capture device; it only rejects non-image types and images outside the 1–4096 px side range. Operators with a fixed camera should validate on frames from that camera.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both. On this repository's smoke run (claude-science WSL venv, RTX 5070 Ti 16 GB, one synthetic 256×256 image through `ConvNeXtClassificationPipeline.from_pretrained().predict`) loading the verified snapshot took 4.67 s and one prediction 1.47 s including transform and first-call CUDA warm-up; the CPU path is exercised only by the unit tests with an injected runner, not by the smoke. Data environment: inputs are assumed to be natural photographs whose subject is one of the 1000 classes, framed roughly as in ImageNet; line drawings, medical scans, satellite tiles, heavy occlusion or unusual viewpoints fall outside that assumption and degrade accuracy in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `top_k_accuracy(predictions, targets, k)` in `pipeline.py`: the fraction of images whose target index appears among the first `k` predicted indices, for any `k` up to the requested `top_k`. It captures discrete correctness of the ranking, which suits a 1000-way single-label classifier where the operational question is "is the right class first, or at least in the shortlist". It says nothing about calibration or per-class behaviour, so a reader using top-1 alone cannot tell whether errors are near-misses (fixable by a shortlist) or confident mistakes. Upstream reports 84.186 % top-1 / 97.124 % top-5 on the ImageNet-1k validation set at 224 px for this checkpoint (upstream README comparison table; the 384-px row is a different checkpoint); this pipeline has not reproduced those numbers and reports no accuracy of its own.

###### Decision thresholds

The default decision rule is `argmax` over the 1000 softmax scores, exposed as `DECISION_RULE = "argmax"` and reported as `predicted_index` / `predicted_label`; this is an implicit threshold of "highest score wins" with no minimum score. No acceptance threshold was set during development and none is shipped: the softmax score is uncalibrated, so any fixed cut-off would be arbitrary. A deployment that needs an abstain option must choose a score cut-off on its own labelled data, trading the cost of a wrong confident label (false positive) against the cost of an unanswered image (false negative) for its application.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; the upstream figures cited above are single validation-set evaluations by the upstream author with no reported interval. Inference is deterministic given the same weights, device and library versions: there is no sampling, stochastic depth and dropout are disabled by `model.eval()`, and no seed is required; small numeric differences between CPU, GPU and cuDNN kernel choices can reorder near-tied classes. The `score` field is a softmax over logits and is not calibrated; a caller who needs probabilities must fit a calibration map (for example temperature scaling) on their own labelled data.

#### Ethical considerations and biases

###### Data

Upstream states the checkpoint was pre-trained on ImageNet-12k, "a 11821 class subset of full ImageNet-22k", and fine-tuned on ImageNet-1k (upstream README `datasets` and "Model Details"); the disclosure stops there — no per-image licensing, consent status or demographic composition is given for either set, and ImageNet is known to contain photographs of identifiable people scraped from the web, so the presence of personal data is not ruled out. This repository distributes code, tests and documentation; the 114 MB `model.safetensors` snapshot is git-ignored and staged locally under `weights/convnext-tiny-in12k/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite and the smoke run in this repository. Where a sensitive use is foreseeable — for example flagging images in a moderation queue — it is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/convnext_classification_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/convnext-tiny-in12k/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; the Hub path is taken only with `allow_download=True` and then through timm's `hf-hub:<id>@<revision>` form; `trust_remote_code` is never enabled (timm executes no remote code). (2) Input integrity — `_validate` rejects non-PIL inputs, empty or over-size batches, and images outside 1–4096 px before the model runs. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, deterministic preprocessing from the snapshot's `pretrained_cfg`, and `model_id`/`model_revision` in every result. (4) Refusals — no feature-map or training API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (class re-balancing) is applied because the pipeline does not train.

###### Risks and harms

Overconfidence out of distribution: an unrelated image still yields a top-1 label with a score that can look high — the smoke run labelled a synthetic colour gradient `analog clock` at score 0.063 — and the operator bears the harm when that label is acted on. Bias in the label space and training images: ImageNet's classes and their examples skew toward Western, web-scraped imagery, so objects common elsewhere are more often mislabelled; data subjects and third parties bear the harm when such labels feed downstream systems. Automation bias: reviewers presented with a confident label check less carefully. Silent preprocessing loss: centre-cropping removes edge content, so a subject near the border can be cropped away. Likelihood under normal photographic use is moderate for the first two and rises sharply off-distribution; magnitude ranges from a wrong tag to a wrongly moderated image.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — its label space cannot do these, and adapting it to try would be a misuse. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating evidence of what an image contains. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's output is a coarse object label.

## Immutable provenance

- Model: `timm/convnext_tiny.in12k_ft_in1k`
- Revision: `aa096f03029c7f0ec052013f64c819b34f8ad790`
- Snapshot manifest: `weights/convnext-tiny-in12k/dimer-base-manifest.json`, `totalBytes` 114390940
- `model.safetensors` SHA-256: `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95` (114374272 bytes)
- `config.json` SHA-256: `7333fa40fddb5a875bcf1bdcfe30916c925c27c6155834b900e2c4cc08dc4333` (662 bytes)
- Weight format: SafeTensors; loader `timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, pretrained_cfg_overlay={"file": ...})`

## Input/output contract

- `ConvNeXtClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `predict(images, top_k=5)` — `images`: one `PIL.Image.Image` or a sequence of 1–64; sides 1–4096 px; any mode (converted to RGB). Returns `{"predictions": [{"predicted_index", "predicted_label", "top_k": [{"label", "index", "score"}, ...]}, ...], "top_k", "decision_rule", "device", "source", "model_id", "model_revision"}`; `score` is the softmax over 1000 classes.
- `top_k_accuracy(predictions, targets, k=1)` — accepts the `predictions` list above or plain index lists.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32 on both CPU and CUDA; preprocessing resize 235 → centre-crop 224, bicubic, ImageNet mean/std from the snapshot `config.json` (`crop_pct` 0.95; the 288-px `test_input_size` is not used).
- Measured (claude-science WSL venv, RTX 5070 Ti, `HF_HUB_OFFLINE=1`): device `cuda:0`, source `local-snapshot`, load 4.67 s, predict 1.47 s, total 6.14 s, top-1 on a synthetic 256×256 gradient image `analog clock` (index 409) at score 0.0634.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. A ConvNet for the 2020s. CVPR 2022. https://arxiv.org/abs/2201.03545
- Wightman. PyTorch Image Models. https://github.com/huggingface/pytorch-image-models (doi:10.5281/zenodo.4414861)
- Deng et al. ImageNet: A large-scale hierarchical image database. CVPR 2009. https://doi.org/10.1109/CVPR.2009.5206848
- Upstream card: https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k
