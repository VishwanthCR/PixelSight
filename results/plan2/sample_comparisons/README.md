# Plan 2 sample comparisons

This folder contains five RGB visual comparisons generated without new model
inference. Each figure compares a completed Sentinel-2 input patch at 10 m
with its corresponding completed LDSR-S2 result at nominal 2.5 m, sampled
with 100 diffusion steps.

Generate or refresh the figures with:

```powershell
python scripts/plan2/generate_ldsr_sample_comparisons.py
```

The results show a 4× output grid. They are qualitative examples, not proof
of true 2.5 m ground detail because no external 2.5 m reference is used.
