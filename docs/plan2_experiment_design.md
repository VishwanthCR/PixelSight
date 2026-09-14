\# PixelSight Plan 2 — Experiment Design



\## 1. Objective



Evaluate a real-world 4× multispectral super-resolution pipeline using

Sentinel-2 Level-2A imagery with four 10 m bands:



\- B02 — Blue

\- B03 — Green

\- B04 — Red

\- B08 — Near Infrared



The target is approximately 2.5 m spatial resolution.



\## 2. Important Ground-Truth Constraint



Sentinel-2 does not provide native 2.5 m imagery.



Therefore, a 2.5 m bicubic upsampled Sentinel-2 image must NOT be treated

as genuine high-resolution ground truth.



Plan 2 will distinguish between:



1\. Controlled benchmark evaluation using native Sentinel-2 imagery.

2\. Genuine high-resolution reference evaluation, if a suitable external

&#x20;  reference dataset is available.



\## 3. Controlled Benchmark



Native Sentinel-2 imagery provides the reference image at 10 m.



A lower-resolution version will be generated through controlled

degradation.



```text

Native 10 m reference

&#x20;       ↓

Controlled degradation

&#x20;       ↓

Lower-resolution LR

&#x20;       ↓

SR model

&#x20;       ↓

Reconstructed 10 m image

&#x20;       ↓

Compare against native 10 m reference

