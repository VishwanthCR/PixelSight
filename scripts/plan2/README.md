# Plan 2: supported workflow

The supported, no-new-inference workflow uses completed 100-step LDSR-S2 patch
outputs and saves all presentation images outside `dataset/`.

```powershell
python scripts/plan2/validate_plan2.py
python scripts/plan2/generate_ldsr_sample_comparisons.py
```

Generated figures are in `results/plan2/sample_comparisons/`.

`visualize_ldsr_result.py` is retained for full-scene 10-step versus 100-step
analysis. It requires both 2048 x 2048 LDSR GeoTIFF outputs in
`dataset/plan2/ldsr_s2_test/`; it will clearly report that prerequisite if the
files are absent. Those files cannot be reconstructed from patch previews.
