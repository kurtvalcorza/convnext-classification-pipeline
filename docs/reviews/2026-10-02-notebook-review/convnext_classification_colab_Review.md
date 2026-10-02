# ConvNeXt-Tiny Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/convnext-classification-pipeline`  
**Notebook:** `tutorials/convnext_classification_colab.ipynb`  
**Reviewed commit:** `826fde7b8bae62acdb001254a7e95a6a565f0628` (`main`, confirmed with `gh api repos/kurtvalcorza/convnext-classification-pipeline/commits/main`)  
**Notebook Git blob:** `79eaced9ad681c4135b2f30ebc45c06f84aa5ca1`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-14 (commit `98fa3d8`, `fetched_blob_verified: true`); `git diff 98fa3d8 826fde7` touches no notebook, generator, template or pipeline source (7 files: documentation, `.gitattributes`, a weight-facts test and `tools/validate_release_assets.py`). Generator `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CNX`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

The inference half is careful and correct. The notebook carries the package module (510 lines, one documented rewrite), asserts the inline manifest against the module identity, stages and re-hashes the pinned `timm/convnext_tiny.in12k_ft_in1k` snapshot, validates a deterministic synthetic gradient into an input manifest with a recorded rejection probe, classifies it with the argmax rule and a rank-ordered top-5, and writes a `not-measurable` evaluation report. Score semantics (uncalibrated softmax, no shipped threshold, caller owns calibration) and the meaning of a label on a gradient are stated correctly.

The fine-tuning half, which the title, objectives and `Run all` paragraph all promise, is where the problems are. A direct CPU run of every code cell at the documented defaults reproduced the Kaggle record in shape:

| Measure | This review (CPU, pinned venv, defaults) | Kaggle T4 record (blob `79eaced9`) |
|---|---|---|
| Code cells completed | 10/10 (9.0 s of cell time; install skipped) | 10/10 on pass 2 (pass 1 stopped at the install guard) |
| Gradient top-1 | `analog clock` 0.0723 | identical |
| Tutorial dataset | `Cleanlab/cifar-10-subset`, 400 images (frog 200 / truck 200, 32×32), digest matches; 16 per class used → 26 train / 6 val | identical split |
| Fine-tune, 1 epoch | train loss 1.2515, val loss 0.0972 | train loss 1.2906, val loss 0.5638 |
| Held-out accuracy (6 images) | **6/6 = 100 %** | **5/6 = 83.3 %** |
| Same split, `SEED` = 0 / 1 / 2 | 83.3 % / 66.7 % / 83.3 % | — |
| `verdict` in the fine-tuned report | `success` in every run | `success` |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CNX-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.1, `numpy` 2.0.2 → 2.5.3) and passed only after a restart; the release record reports it as "PASSED — 10/10".
2. **"Classification head fine-tuning" trains the whole network, and the run's configuration is not recorded (CNX-M2).** `fit` hands `model.parameters()` to AdamW; all 180 floating-point tensors (27,820,128 parameters) of the exported artifact differ from the base snapshot. Learning rate, batch size, weight decay, seed, dataset digest and the trainable set are not in `result.json` or `model-config.json`.
3. **The fine-tune evaluation always says `success` on six images (CNX-M3).** The verdict is a literal. Accuracy moves between 66.7 % and 100 % with the split seed and the device; the 95 % Wilson interval for the recorded 5/6 is [0.44, 0.97], which contains the 50 % majority baseline. A BYOD run that scored exactly the baseline was also labelled `success`. Nothing tells the learner these are six-image tutorial numbers.
4. **Guided layer largely absent (CNX-M4).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, task contract, glossary, prediction, checkpoint, troubleshooting or conclusion template, no rerun guidance for the suggested experiments, and the 510-line carried module is not labelled as infrastructure.

The air-gapped fallback that the notebook promises crashes with a `NameError` (CNX-m1), and the REL12 BYOD release gate has not been run on a hosted runtime; this review exercised both BYOD branches locally (§4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL image handling; what a softmax over class logits is" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA used when available; float32 |
| Promised outcomes | Pinned install; carried module; digest-verified snapshot; synthetic sample → input manifest → argmax + top-5 → `not-measurable` report; digest-verified `Cleanlab/cifar-10-subset` with a seeded 80/20 split; "100% in-kernel classification head fine-tuning" with AdamW and cross-entropy; export `model.safetensors` + `model-config.json`; fresh-boundary reload "to verify artifact integrity"; held-out evaluation against a majority baseline; outputs and provenance; air-gapped fallback to "deterministic synthetic stripes"; BYOD single image (with optional ImageNet index) and BYOD dataset zip through "the same validation, seeded split, in-kernel fine-tuning, export, fresh-reload and held-out evaluation cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `216da18` |

### Evidence actually obtained

- **Source inspection.** All 23 cells (10 code; cell 5 is the carried `pipeline.py`). Also read: `pipeline.py` (`from_pretrained`, `fit`, `predict`, `_check_inputs`, `validate_inputs`, `evaluation_report`), the generator and template, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `dimer-nb2-convnext-classification` v1 (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`). Kaggle Tesla T4, 2026-09-14, **the reviewed blob**, clean HF cache, image torch 2.10.0+cu128 / numpy 2.0.2. Pass 1 failed in cell 3 with the restart `RuntimeError` after 150.6 s; pass 2 ran 10/10 in 28.0 s. No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, in the build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, torchaudio 2.11.0, timm 1.0.29, numpy 2.5.3, pillow 11.3.0, safetensors 0.8.0, huggingface-hub 0.36.2 — the notebook's `PINS`). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Not a clean runtime:** the three snapshot files were hard-linked into a scratch working directory; cell 7 wrote its manifest fresh and `verify_snapshot` re-hashed every file. The tutorial dataset was fetched by cell 17 itself from its pinned URL.
  - **Executed:** every code cell at defaults (P1); dataset facts (P2); reload equivalence and parameter-change count (P3); cell 17 + 19 with `SEED` 0, 1, 2 (P4); cell 17 with the dataset URL unreachable (P5); BYOD single image with `GROUND_TRUTH_INDEX`, an out-of-range index and a non-image file (P6); BYOD dataset branch with three accepted and six rejected/failing archives (P7). `google.colab.files.upload` was replaced by a shim returning prepared bytes; the Colab upload dialog itself was not exercised.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good on the default path (P1 10/10; reload reproduces the in-memory model exactly, 6/6 same labels, max score difference 0.0). Defects: the install pattern forces a restart (CNX-M1); the documented air-gapped fallback references two undefined names (CNX-m1); BYOD dataset checks run after training or not at all (CNX-m2).
- **Promise fulfilment:** inference promises met. "Head fine-tuning" is not what runs (CNX-M2); "verify artifact integrity" is a load without a comparison (CNX-m3); "gracefully falls back" crashes (CNX-m1).
- **Scientific validity:** the fine-tune evaluation is a six-image tutorial check presented as `success` regardless of outcome (CNX-M3).
- **Learner experience:** clear inference prose with three "look for" notes; the fine-tuning sections have none, and the guided layer is mostly missing (CNX-M4). Several statements contradict the run (CNX-m4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6 (CNX-M1); FT3, FT5, FT6, OUT8 (CNX-M2); EVAL6, DAT8 (CNX-M3); DAT19, VAL1 (CNX-m2); VER5 (CNX-m3); SRC3 (CNX-m4); REL12 BYOD evidence absent from the release record. SHOULD deviations: GDL1–GDL4, GDL6, GDL7, GDL9–GDL14, UX8 (CNX-M4); VER4 (CNX-m3); OUT9 (CNX-M2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 10/10 | Section 1 says the cell "stops with a restart instruction" | **Not met** (CNX-M1) |
| Digest-verified pinned snapshot | cell 7 | 3/3 files verified, revision `aa096f03…` | clear | Met |
| Synthetic sample → input manifest with rejection probe | cells 9, 11 | `accepted`; oversized probe rejected with the ceiling named | clear | Met |
| Argmax + rank-ordered top-5, uncalibrated scores | cell 13 | `analog clock` 0.0723, flat top-5 | "expect a low top-1 score spread across unrelated classes" | Met |
| Evaluation report `not-measurable` on the gradient | cell 15 | verdict and `needs` printed | clear | Met |
| Digest-verified tutorial dataset, seeded split | cell 17 | 400-image zip, digest matches; 26/6 | "80/20" (actual 81/19 after the 16-per-class subset); 32 px images not mentioned | Met |
| Air-gapped fallback to synthetic stripes | cell 17 `else` branch | `NameError: name 'SAMPLE_HEIGHT' is not defined` after the "falling back" warning | — | **Not met** (CNX-m1) |
| "100% in-kernel classification head fine-tuning" | cell 17 → `fit` | all 27.8 M parameters updated | prose says head only | **Not met** (CNX-M2) |
| Export + fresh-boundary reload "to verify artifact integrity" | cell 19 | loads with `strict=True`; no comparison with `fine_tuned_pipe` | "verify" | Partly met (CNX-m3) |
| Held-out evaluation against majority baseline | cell 19 | 5/6 (T4), 6/6 (CPU); baseline 50 %; verdict `success` | "success", no uncertainty, not labelled tutorial evidence | Met as computation, misleading as conclusion (CNX-M3) |
| Outputs and provenance | cell 21 | 6 files + artifact dir; no fine-tune hyperparameters or dataset digest | listed | Partly met (CNX-M2) |
| BYOD single image, optional ImageNet index | cell 9 | P6: report switches to `sample-sanity`, k=1/k=5 printed; index 1000 refused with the range named; non-image upload raises a bare `UnidentifiedImageError` | contract stated in Prerequisites | Met (local), recovery message weak (CNX-m2) |
| BYOD dataset zip, "same validation … cells" | cell 17 | P7: accepted archives reach fit → export → reload → evaluate; image-side ceiling enforced only after training; mislabelled layouts accepted silently | zip rules partly stated in Section 8 | Partly met (CNX-m2) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried module, verify the revision | run cells | printed versions and verified-file count |
| Validate a synthetic input; read argmax and uncalibrated top-k | run cells, read outputs | outputs readable; one "look for" hint; no prediction asked |
| "Execute 100% in-kernel fine-tuning on custom classes" | run cell | runs; what was trained is misdescribed (CNX-M2) |
| Export and fresh-reload fine-tuned artifacts | run cell | load only (CNX-m3) |
| Exercise an optional BYOD path | change a field, upload | possible; no rerun guidance (CNX-M4) |
| Evaluation report semantics | run cell | `not-measurable` / `sample-sanity` explained; fine-tune report not explained (CNX-M3) |

Objectives are phrased as actions the code performs (GDL5), and none is followed by a check of the learner's understanding.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 23 cells | Inference sections are accurate and give "look for" hints. Section 8 introduces AdamW, cross-entropy, head replacement and a balanced subset with no expected-result note; Section 9 prints a verdict of `success` with no guidance on six-image uncertainty (CNX-M3); the interpretation section does not mention the fine-tune result at all. Prerequisites say "nothing is downloaded" and list the model snapshot as the only external access, then Section 8 downloads a dataset (CNX-m4). No audience, roadmap, glossary, predictions, checkpoints or troubleshooting (CNX-M4). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard after 150.6 s, pass 2 10/10 after a restart (CNX-M1). Direct: 10/10 at defaults, numbers in the table above; six output files plus the two-file artifact directory written; reload equivalent. No Colab run. |
| **Active learning** | Direct (P4, P6) | "Next experiments" item 1: `USE_BYOD = True`, `GROUND_TRUTH_INDEX = 867`, re-ran cells 9–15 → report switched to `sample-sanity` with `top_k_accuracy` 0.0 at k=1 and k=5 (top-1 `moving van` on an upscaled CIFAR truck; labelled stand-in). The notebook does not say which cells to re-run or that re-running onward repeats the fine-tune. Changing the split seed (a plain variable, not a field) moved held-out accuracy 66.7–83.3 % (CNX-M3). Epochs, learning rate and batch size are literals inside the `fit` call, so the notebook offers no fine-tuning experiment (CNX-M4). |
| **Reuse and recovery** | Direct (P5–P7); Colab upload dialog not verified | Dataset URL unreachable: "falling back to deterministic synthetic dataset" then `NameError` (CNX-m1). BYOD dataset: two-class folder archive accepted and carried through fit → export → reload → evaluate (accuracy 0.5 = baseline, verdict `success`). Refused with a named rule: one class, `../` path, class with one image, `.tar` upload. Accepted silently: a `val/` class absent from `train/` (2 of 4 validation images dropped), images at the archive root (became class `unknown`). Failed late: an image wider than 4096 px trained, then cell 19 raised the ceiling error. Failed raw: an undecodable image raised `UnidentifiedImageError` without naming the file (CNX-m2). Single image: index 1000 refused with the range; a text file named `.png` raised a bare `UnidentifiedImageError`. |

## 5. Findings

### Major

#### CNX-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, install block lines 48–70); `docs/release-verification.md` record and procedure.
- **Observed issue:** the cell `pip install`s eight pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. Section 1 prose presents this as expected behaviour.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle/Colab image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this.
- **Evidence:** documented — Kaggle T4 run of blob `79eaced9`, pass 1 `ok: false` with `cuda-bindings: loaded=12.9.4, installed=13.4.1; numpy: loaded=2.0.2, installed=2.5.3`, pass 2 after restart 10/10; the record row reads "PASSED — 10/10 ok code cells executed cleanly" with no mention of the restart. Source — `pip_install_in_kernel: true`, `uses_uv: false` (probe static block).
- **Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/convnext_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### CNX-M2 — "Classification head fine-tuning" updates the whole network, and the adaptation configuration is not recorded

- **Cell/section:** opening cell ("100% in-kernel classification head fine-tuning", "head adaptation via `fit`"), Section 8 prose ("implements 100% in-kernel head adaptation"), cell 17 `fit` call, cell 21 export; `pipeline.py` `fit`. Generator: `tools/notebook_template.py` lines 66–68, 209, 343; `src/convnext_classification_pipeline/pipeline.py` `fit` (`AdamW(model.parameters(), …)`) and its `config_payload`.
- **Observed issue:** `fit` builds a new 2-class head and optimises **all** parameters. The notebook never states which parameters are trainable. Epochs (1), batch size (4), learning rate (1e-4) are literals in the call; weight decay (0.01) and the training seed (20260910) are `fit` defaults that the learner never sees. `result.json` records only `history`, `classes` and the evaluation; `model-config.json` records architecture, classes, model identity and `data_config`. Neither records optimiser settings, seed, trainable set, dataset source/digest or split; `dataset_source` is printed but not exported.
- **Consequence:** the learner is taught that only a linear head was trained when the backbone moved too, which changes what the result means (cost, overfitting risk on 26 images, what transfers). The exported artifact cannot be interpreted or reproduced from its own provenance.
- **Evidence:** direct (P3) — exported `model.safetensors` vs base snapshot: 180/180 shared floating-point tensors changed, 27,820,128/27,820,128 parameters, including `stages.0.blocks.0.conv_dw.bias`; source — probe `fit_optimizer_line`, `result_json_finetune_keys`.
- **Recommended correction:** decide the method, then make text and code agree. Either freeze the backbone in `fit` (a `train_backbone=False` default, trainable-parameter count printed) so "head fine-tuning" is true, or describe it as full fine-tuning everywhere. Lift `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `TRAIN_SEED` and the trainable set into form fields in cell 17, print them before training, and write them — with the dataset source, zip SHA-256, class list and split sizes — into both `result.json` (`fine_tuning.config`) and `model-config.json`.
- **Acceptance check:** the notebook prints the number of trainable and frozen parameters before training, and the description in the opening cell and Section 8 matches that count; `outputs/convnext_classification_result.json` and `outputs/convnext_classification_finetuned/model-config.json` both contain `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `seed`, `trainable`, `dataset_sha256` and split sizes.
- **Spec:** FT3, FT5, FT6, FT4, OUT8, OUT9.

#### CNX-M3 — The fine-tuned evaluation always reports `success`, on six held-out images with no uncertainty

- **Cell/section:** cell 19 (`'verdict': 'success'`), Section 9 prose, Interpretation section. Generator: `tools/notebook_template.py` line 400 and the interpretation block from line 482.
- **Observed issue:** the verdict is a string literal, independent of the measured accuracy. The held-out split is 6 images (3 per class, 32×32 CIFAR crops upscaled to 224); one image moves accuracy by 16.7 points. Nothing labels the result as a tutorial metric, states the estimation procedure's size limit, or shows which images failed; the interpretation section discusses only the ImageNet classifier and never interprets the fine-tune.
- **Consequence:** the learner is told the adaptation "succeeded" whatever happens, and is likely to read "83 % vs 50 % baseline, +33 %" as a demonstrated improvement. With 6 images that conclusion is not supported: the 95 % Wilson interval for 5/6 is [0.44, 0.97] and contains the baseline. A run that scores exactly the baseline is also labelled `success`.
- **Evidence:** documented — Kaggle 5/6, verdict `success`. Direct — P1 CPU 6/6; P4 split seeds 0/1/2 → 83.3 / 66.7 / 83.3 %, all `success`; P7 BYOD archive accuracy 0.5 = baseline 0.5, verdict `success`; source — probe `verdict_success_hardcoded: true`.
- **Recommended correction:** derive the verdict from the evidence (e.g. `sample-sanity` with the accuracy, its Wilson interval and the baseline; never `success` when the interval includes the baseline), label the block "tutorial metric on N held-out images, not a benchmark", add per-class counts and the misclassified images to the printed output, and add an interpretation paragraph (or a conclusion template) for the fine-tune result that names the sample size. Optionally raise `SUBSET_PER_CLASS` so the held-out split is larger.
- **Acceptance check:** `grep -n "'verdict': 'success'" tutorials/convnext_classification_colab.ipynb` returns nothing; the fine-tuned report and the printed summary include `n`, a confidence interval and the baseline; a run whose accuracy equals the baseline does not print a success verdict; the interpretation section mentions the fine-tune result and its sample size.
- **Spec:** EVAL6, DAT8, EVAL3, EVAL5, EVAL15, RUN8.

#### CNX-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cell 5, end of notebook. Generator: `tools/notebook_template.py`.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output contract (two tasks — ImageNet inference and 2-class adaptation — share one notebook), no glossary (ConvNeXt, logits/softmax, AdamW, cross-entropy, head replacement, majority baseline, center crop), no prediction before the classify, fine-tune or evaluation results, no interpretation checkpoint, no troubleshooting section (download failure, out-of-memory, BYOD errors, the restart in CNX-M1), no conclusion template; three "look for" notes, none in Sections 8–9. The "Next experiments" do not say which cells to re-run, and the second one ("supply your own multi-class dataset folder to `pipe.fit`") does not point to the `USE_BYOD_DATASET` field that already exists. No fine-tuning control is a form field. The 510-line carried module in cell 5 is not labelled **Infrastructure** or collapsed (`cellView` absent).
- **Consequence:** a self-paced learner gets an accurate script but little help deciding what matters, what normal output looks like in the fine-tune stages, or how to run a valid experiment.
- **Evidence:** source inspection; probe `guided_markers` (How to use / Roadmap / Glossary / Check your reasoning / What to notice / Troubleshooting / conclusion / audience / Infrastructure / rerun all absent), `cellView_form_cells: []`, `fit_call_hyperparams_are_form_fields: false`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC §25.13's reference notebook: audience and how-to-use, roadmap, task contracts for both capabilities, glossary, a prediction before Sections 6, 8 and 9, "What to notice" after each principal stage, collapsible checkpoint answers, a Predict → Change one thing → Run → Observe → Explain activity (for example head-only vs full fine-tuning, built on CNX-M2's fields) with an explicit "re-run cells N–M" instruction, troubleshooting, and a conclusion scaffold; title cells 3, 5 and 7 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL7, GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`; cells 3, 5 and 7 carry `cellView: form` with an Infrastructure title; every "Next experiments" item names the field to change and the cells to re-run.
- **Spec:** GDL1–GDL4, GDL6, GDL7, GDL9–GDL14, UX5, UX8.

### Minor

#### CNX-m1 — The promised air-gapped fallback crashes with `NameError`

- **Cell/section:** cell 17 `else` branch; Section 8 prose ("it gracefully falls back to deterministic synthetic stripes"). Generator: `tools/notebook_template.py` lines 216–217 and 322.
- **Observed issue:** the fallback builds arrays of shape `(SAMPLE_HEIGHT, SAMPLE_WIDTH, 3)`; neither name is defined anywhere in the notebook or the carried module (cell 9 defines `SAMPLE_SIDE`).
- **Consequence:** if the dataset download fails (Hub outage, firewalled runtime), the learner sees "falling back to deterministic synthetic dataset" and then a `NameError`; `Run all` stops at Section 8 and nothing after it runs.
- **Evidence:** direct (P5) — URL pointed at an unreachable host: warning printed, then `NameError: name 'SAMPLE_HEIGHT' is not defined`; static probe `fallback_names_undefined: ['SAMPLE_HEIGHT', 'SAMPLE_WIDTH']`. The CI parity and compile checks cannot see this because the branch is never executed.
- **Recommended correction:** use `SAMPLE_SIDE` (or define the two names), and add a unit test in `tests/test_notebook_parity.py` (or a new test) that execs the fallback branch with the download forced to fail. Alternatively drop the fallback and fail with an actionable message naming the URL and digest.
- **Acceptance check:** with `SAMPLE_DATASET_URL` unreachable, cell 17 completes and cell 19 runs on the fallback data, or cell 17 stops with a message naming the URL and the next step; a test covers the branch.
- **Spec:** SRC2, UX10, RUN9.

#### CNX-m2 — BYOD dataset branch: checks run late or not at all, and some layouts are mislabelled silently

- **Cell/section:** cell 17 BYOD branch; cell 9 BYOD image; Prerequisites (`Data:` bullet).
- **Observed issue:** the dataset images never pass through `validate_inputs`, so the stated 4096 px ceiling is enforced only by `predict` in cell 19, after training. A `val/` class absent from `train/` is dropped without a message; images at the archive root become a class named `unknown`; an undecodable file raises a bare `UnidentifiedImageError` without naming it; the same for a non-image single-image upload. No limits on archive size, image count or classes are stated, and the Prerequisites describe only the single-image upload.
- **Consequence:** a user who brings their own data can train on silently mislabelled or truncated data, or wait through a fine-tune only to fail in evaluation, with errors that do not say which file to fix.
- **Evidence:** direct (P6, P7) — `b_trainval_val_class_missing_from_train`: val 2 instead of 4, no message; `c_root_level_images`: classes `['blue', 'unknown']`; `g_oversized_image_in_val`: cell 17 ok, cell 19 `ValueError: image side outside 1..MAX_IMAGE_SIDE=4096 px: (4100, 8)`; `h_corrupt_image` and the single-image text file: `UnidentifiedImageError: cannot identify image file <_io.BytesIO …>`. Refused correctly with the rule named: one class, `../` path, one image in a class, `.tar` upload, `GROUND_TRUTH_INDEX = 1000`.
- **Recommended correction:** before `fit`, run every decoded dataset image through `validate_inputs` and wrap decoding to name the archive member; reject (or report) val classes missing from train and files outside a class folder; state the zip layout, per-class minimums, size limits and a privacy note in the Prerequisites; print a dataset manifest (classes, per-class counts, dropped files and why).
- **Acceptance check:** each of the four archives above (`val` class missing from `train`, root-level images, a 4100 px image, an undecodable member) is refused in cell 17, before any training, with a message naming the file or class and the rule; the Prerequisites state the zip contract.
- **Spec:** DAT12, DAT13, DAT19, VAL1, VAL2, VAL7, UX10, REL12.

#### CNX-m3 — "Verify artifact integrity" is a load, not a comparison

- **Cell/section:** Section 9 prose and cell 19. Generator: `tools/notebook_template.py` lines 362–373.
- **Observed issue:** cell 19 reloads the artifact with `strict=True` and evaluates it, but never compares its outputs with the in-memory `fine_tuned_pipe` (used nowhere after cell 17) and does not hash the exported files. The prose says the reload "verif[ies] artifact integrity".
- **Consequence:** the learner cannot tell a successful load from an equivalent model; a serialisation regression that still loads would pass silently.
- **Evidence:** source — `fine_tuned_pipe_uses: 1`; direct (P3) — the comparison the notebook omits holds: 6/6 identical labels, max top-k score difference 0.0.
- **Recommended correction:** predict the held-out images with both pipelines, report label agreement and the maximum score difference against a stated tolerance, record the artifact's SHA-256 in `result.json`, and say in prose that loading alone is not the check.
- **Acceptance check:** cell 19 prints `equivalent: True` (or the failing difference) with a tolerance, and `result.json` contains the artifact digest and the comparison.
- **Spec:** VER4, VER5, REL7.

#### CNX-m4 — Statements that contradict the run or each other

- **Cell/section:** cell 1 Prerequisites; Section 8 prose; `docs/release-verification.md`; `STATUS.md`.
- **Observed issue:** Prerequisites say "nothing is downloaded" and name the model snapshot as the only external access, yet Section 8 downloads a 986,707-byte dataset; the knowledge prerequisites omit fine-tuning concepts; no run time is given although the record measured 178.7 s on T4. The release record's executor table names a "Kaggle CPU kernel" while the run was on a T4; its procedure (step 5) lists no fine-tuning, reload or held-out evaluation stage; its "Current status" paragraph begins "No clean-runtime execution of the notebook has been recorded yet" under a recorded row. `STATUS.md` still says "awaiting clean-runtime execution" while `tutorials/README.md` says "verified".
- **Consequence:** the learner plans for the wrong network access and time; a release reviewer cannot tell which stages the recorded run was meant to cover.
- **Evidence:** source inspection; documented record (pass 1 150.6 s + pass 2 28.0 s); probe `prereq_nothing_downloaded_claim: true`.
- **Recommended correction:** correct the Prerequisites (dataset download, its source and size, fine-tuning concepts, measured time with runtime and date); update the release procedure to cover Sections 8–9 and record the executor as T4; make `STATUS.md`, `tutorials/README.md` and the record agree.
- **Acceptance check:** the Prerequisites name both downloads; `docs/release-verification.md` step 5 lists the fine-tune, reload and held-out evaluation stages; `STATUS.md`, `tutorials/README.md` and the record carry one consistent status.
- **Spec:** SRC3, UX12, REL10.

### Suggestions

- **CNX-S1** — Declare `notebook_spec` 2.2 instead of 2.0 once the guided layer lands.
- **CNX-S2** — Show the six (or more) held-out images with true label, predicted label and score, so the learner can see what the errors look like.
- **CNX-S3** — Offer head-only vs full fine-tuning as the guided experiment (it falls out of CNX-M2's correction) and print both results side by side.
- **CNX-S4** — Record per-stage wall times in `result.json`, and state in Section 8 that the CIFAR images are 32×32 and upscaled to 224 px.

## 6. Readiness

**Needs revision.** Open Majors CNX-M1 to CNX-M4. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10), the REL12 BYOD exercise recorded in `docs/release-verification.md` (one compatible archive, one incompatible archive), and a release record whose procedure covers the fine-tune, reload and evaluation stages.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, labelled above):** default path 10/10; full-network parameter updates; reload equivalence; held-out accuracy 66.7–100 % across seeds and devices with a constant `success` verdict; the fallback `NameError`; the BYOD image and dataset outcomes listed in §4.
- **Verified from documented evidence:** the restart on Kaggle pass 1, the pass-2 numbers, the 5/6 held-out result on T4.
- **Inferred from source:** that the Colab upload dialog delivers files as the shim did; that GPU runs vary as the CPU seed sweep did (only one T4 run exists).
- **Only Kurt can confirm:** whether the intended method is head-only or full fine-tuning (CNX-M2 accepts either, provided text and code agree).
- **Most likely to be wrong:** CNX-m1's severity — the fallback is a documented recovery path whose failure stops `Run all`, so a maintainer could rate it Major; I rated it Minor because it triggers only when the pinned Hub download fails.
