# Weight provenance and DIMER hosting

- Upstream: `timm/convnext_tiny.in12k_ft_in1k`
- Immutable revision: `aa096f03029c7f0ec052013f64c819b34f8ad790`
- Weight format: SafeTensors (`model.safetensors`, 114374272 bytes)
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/convnext-tiny-in12k/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 114390940); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/convnext_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `timm==1.0.29` built-in `convnext_tiny` architecture; weights loaded from a file path via `pretrained_cfg_overlay`; no remote code is executed. Hub download is opt-in and pinned to the revision above.
