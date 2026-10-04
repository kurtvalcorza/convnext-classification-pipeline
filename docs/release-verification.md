# Release verification

`tutorials/convnext_classification_colab.ipynb` (`E2E`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, `standalone: true` and `generated_from` (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/convnext_classification_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; exactly two kernel cells — the isolated install (pinned `uv` wheel checked by size and SHA-256, managed
  CPython, `--require-hashes --only-binary :all:` from the carried hash lock) and the router — with every later cell
  routed to the isolated environment; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ConvNeXtClassificationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `predict`, `evaluation_report`), the ceiling print (`NUM_CLASSES`, `MAX_IMAGE_SIDE`, `MAX_BATCH`),
  the exports, the learner-facing classification statements (argmax decision rule, uncalibrated
  softmax, no shipped threshold, rank-ordered scores) and the gated-off BYOD default listed in
  the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `timm.create_model` / `from timm import` / `from torchvision import` /
  `from transformers import` / `from huggingface_hub import` use **outside the carried module cell**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- the review fixes (CNX-M1..M4, CNX-m1..m4): the fine-tuning configuration, trainable set and dataset provenance are
  printed and exported, the held-out verdict comes from `finetune_evaluation_report` (no literal verdict), the
  fallback and BYOD archives go through `synthetic_stripes_dataset` / `load_image_zip`, the reload is compared with
  the in-memory model, stale learner-facing claims are absent, and the guided layer (who it is for, how to use,
  roadmap, predictions, *What to notice*, worked answers, the Section 12 activity, troubleshooting, glossary,
  conclusion; Infrastructure titles on the five setup cells) is present;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchvision` wheels plus `timm`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab runtime, Linux x86_64 (T4 GPU or CPU; CUDA used automatically when present) | The runtime the tutorial is written for; a one-pass top-to-bottom `Run all` here is promotion evidence |
| Colab CLI / Kaggle kernel | Fresh Linux x86_64 VM (Tesla T4) | Clean-room executor of the same class; the notebook is executed verbatim (Kaggle: plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory). No repository checkout is needed — the notebook is standalone |
| Local harness (pre-flight only) | Workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` and a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

The notebook supports **Linux x86_64 runtimes only**: its isolated environment is built from manylinux wheels, and
Section 1 stops with that message on Windows or macOS.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new Linux x86_64 runtime (Colab, or the executors above) with **no
   repository checkout** and a clean model cache;
3. choose **Run all once** without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`, `USE_BYOD_DATASET = False`, `TRAINABLE = 'head'`); a run that needs
   a restart or a second pass is not a `Run all` PASS and must be recorded as such;
4. verify that Section 1 builds the isolated environment (`isolated_python` 3.12.12, the locked package count) and
   that the runtime cell reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and the installed core package versions equal to the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - pinned `timm/convnext_tiny.in12k_ft_in1k` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/convnext-tiny-in12k/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - synthetic 256×256 gradient sample generated in code with its RGB SHA-256 printed and the
     ceilings (`NUM_CLASSES` 1000, `MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 64) surfaced;
   - `validate_inputs` writes `outputs/convnext_classification_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the oversized probe);
   - classification through `predict(image, top_k=5)` with `decision_rule == 'argmax'` and a
     rank-ordered top-5 list;
   - `evaluation_report` writes `outputs/convnext_classification_evaluation_report.json` with verdict `not-measurable`
     on the synthetic sample (no ground truth), stated as such;
   - Section 8: the tutorial dataset is downloaded and digest-verified (`66f90a4f…`, not the synthetic fallback), with
     a pair-grouped split of 80 training and 20 held-out images per class (both copies of a photo on one side) and
     `held_out_sharing_a_file_name_with_train: 0`;
   - Section 9: `head-only fine-tuning` with 1,538 trainable and 27,820,128 frozen parameters printed before training,
     and the one-epoch history;
   - Section 10: the reloaded artifact reports `source == 'fine-tuned-artifact'` and `equivalent: True`, and the
     held-out report prints `n = 40`, the count, the 95 % Wilson interval, the majority baseline, the verdict
     `sample-sanity` and `comparison_to_baseline`;
   - `outputs/convnext_classification_result.json` (with `fine_tuning.config`, `fine_tuning.dataset`,
     `fine_tuning.evaluation` and `fine_tuning.artifact_sha256`), `outputs/convnext_classification_top_k.csv`,
     `outputs/convnext_classification_validation_predictions.csv`,
     `outputs/convnext_classification_finetuned_evaluation_report.json` and
     `outputs/convnext_classification_finetuned/{model.safetensors,model-config.json}` (with `fine_tuning` in the
     config) are written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. exercise the BYOD gates (REL12): one compatible and one incompatible dataset archive through Section 8 (the
   incompatible one must be refused before training with the file or class named), and optionally one BYOD image;
8. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device),
   model identifier and immutable revision, whether the model cache was clean, `restarted: false`, outcome, the
   held-out count with its interval, produced outputs, and any warning or applicable `SHOULD` deviation in the table
   below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/convnext_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/convnext_classification_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `98fa3d8` / `79eaced9ad68` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-convnext-classification` v1) | Default path (inference, fine-tuning, reload, held-out evaluation) | pass 1 150.6 s + pass 2 28.0 s | **Not a one-pass `Run all`; not promotion evidence.** Pass 1 stopped in the install cell with the restart `RuntimeError` (`cuda-bindings` 12.9.4 → 13.4.1, `numpy` 2.0.2 → 2.5.3); pass 2, after the restart, ran 10/10 code cells (held-out 5/6, verdict then hard-coded `success`). Superseded by the review fixes (isolated environment, no restart); the current blob has no hosted run yet. |
| 2026-10-04 | `625fd1b` / `1f059ceaa746` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-convnext-625fd1b-a531`) | Default path only (`TRAINABLE = 'head'`; inference, head-only fine-tuning, reload, held-out evaluation) | 106.0 s | **One pass, no restart, 0 errors**, 14/14 code cells. Held-out **37/40 = 92.5 %, 95 % Wilson 80.1 %–97.4 %**, above the 50 % majority baseline (verdict `sample-sanity`, `above-baseline`). See the record below. |

### 2026-10-04 — Colab CLI one-pass run of `625fd1b` (fresh Colab Tesla T4)

- **Commit / notebook blob:** `625fd1b543f13020dcf438574c7f6d5672c293c7` / `1f059ceaa7468cb857bce1fdff945419a60c9f6b`
  (blob checked against the fetched bytes before the VM was allocated; executed code-cell sources equal the commit's).
- **Executor:** Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab VM, Tesla T4, no repository
  checkout, clean model cache. This is **not** a browser `Run all`: the CLI sets no execution counts, so cell order is
  evidenced by `exec.log` (`Executing cell 1/14` … `14/14`, in order); forms were not rendered.
- **Path:** default settings only (`USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`, `USE_BYOD_DATASET = False`,
  `TRAINABLE = 'head'`). Wall time 106.0 s (suite wall clock, including installs and the model download).
- **Outcome:** **one pass, no restart, 0 errors**; 14/14 code cells; cell 4 (the carried module definition) prints
  nothing by design. One Pillow `DeprecationWarning` (`mode` parameter) in the sample cell.
- **Runtime:** isolated Python 3.12.12 (kernel 3.13.15), 45 locked packages, torch 2.14.0+cu130, timm 1.0.29,
  `cuda: True`, device `cuda:0`; `NOTEBOOK_SOURCE.repository_revision` `d31f2ee1` = `metadata.dimer.generated_from`.
- **Model:** `timm/convnext_tiny.in12k_ft_in1k` @ `aa096f03029c7f0ec052013f64c819b34f8ad790` (apache-2.0, 3 files,
  114,390,940 bytes), loaded with `source == 'local-snapshot'`.
- **Inference:** synthetic 256×256 gradient (RGB SHA-256 `e38db00b…`), input manifest `accepted` with the oversized
  probe `rejected`; `decision_rule == 'argmax'`, top-1 `analog clock` (409); evaluation report `not-measurable`
  (no ground truth).
- **Section 8:** tutorial dataset `Cleanlab/cifar-10-subset @ bb5a7aab`, SHA-256 `66f90a4f…`; pair-grouped split
  80 train / 20 held out per class (160 / 40), `held_out_sharing_a_file_name_with_train: 0`.
- **Section 9:** head-only fine-tuning, 1,538 trainable / 27,820,128 frozen parameters, 1 epoch, batch 4,
  lr 1e-4; train loss 0.4082, val loss 0.2318, val accuracy 0.925.
- **Section 10:** reload `source == 'fine-tuned-artifact'`, 40/40 label agreement, max score difference 0.0,
  `equivalent: True`. Held-out **37/40 = 92.5 %**, 95 % Wilson interval **80.1 % to 97.4 %**, majority baseline
  50.0 % (`frog`), verdict `sample-sanity`, `comparison_to_baseline: above-baseline`; frog 19/20, truck 18/20;
  misclassified: `original_images/frog/image_9.png` and both copies of `truck/image_13.png`. These equal the
  worked answers' local CPU figures (loss 0.41 / 0.23, 37/40, interval 80–97 %).
- **Exports:** the result JSON, top-k CSV, validation-predictions CSV, input manifest, both evaluation reports and
  `convnext_classification_finetuned/{model.safetensors,model-config.json}` were written.
- **Evidence files** (`docs/execution-evidence/2026-10-04/`, byte-for-byte from the run directory):
  - `convnext_classification_colab_625fd1b_colab-cli-t4_output.ipynb` — SHA-256 `da6df13de56ea34169d3683fedf6220f1c7cd6041149e5f84317c31bdddee2c6`
  - `exec.log` — SHA-256 `8e97434202fc7599d30378dbb74fe6a78d89932622cc172abdf5cd528a082329`
  - `run_summary.json` — SHA-256 `19356881291de881b7dd8f5c36d40a3d4714b176a066365744488c64aac3be67`
- **Not exercised:** the BYOD image and BYOD dataset gates (step 7), the Section 12 activity (`TRAINABLE = 'all'`)
  and any other optional journey; no browser interaction.

## Current status

**Candidate.** The review fixes (CNX-M1..M4, CNX-m1..m4, review PR #8) regenerated the notebook: no in-kernel
install (the fleet's uv isolated environment), the fine-tuning method stated as it runs and its configuration
exported, a held-out verdict computed from the counts with its interval, a validated dataset intake, a working
fallback, a reload equivalence check and the guided layer. Static validation, the generator `--check`, ruff and the
offline unit suite pass on this source. A one-pass hosted run of the current notebook blob `1f059cea` is recorded
above (2026-10-04, Colab CLI sequential execution on a fresh Tesla T4, default path, 14/14 cells, no restart,
held-out 37/40, Wilson 80.1 %–97.4 %). Remaining gates: the BYOD release gate (step 7) on a hosted runtime and a
reviewer's confirmation of the recorded run against the blob under review. The registry status remains
**Candidate** until an integrator promotes it; promotion is not performed by the builder.
