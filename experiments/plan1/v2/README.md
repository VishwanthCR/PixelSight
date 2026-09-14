# Plan 1 — SwinIR v2 Experiment

## Overview

SwinIR v2 is a residual super-resolution architecture designed to improve the
Plan 1 ×2 multispectral super-resolution pipeline while preventing
hallucination on empty patches.

### Architecture

```text
LR 32×32×4
      │
      ├── Bicubic ×2 ──────────┐
      │                        │
      └── SwinIR residual ─────┤
                               +
                               │
                         SR 64×64×4