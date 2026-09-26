# PixelSight Analyst: Local LLM Interpretation Layer

## Overview

**PixelSight Analyst** is a dedicated local language model interpretation layer for the PixelSight satellite super-resolution framework. It provides natural-language scientific explanations, metric interpretation, and contextual question-answering directly on top of completed PixelSight processing jobs.

```
Existing PixelSight Application
            │
            ▼
   Completed Job Results
            │
            ▼
     Context Builder
            │
            ▼
 Scientific Context JSON
            │
            ▼
      Ollama Runtime
            │
            ▼
   Local Instruct LLM
 (e.g. LLaMA 3.1 8B Instruct)
            │
            ▼
   PixelSight Analyst
            │
            ▼
          User
```

---

## Core Scientific Principles

1. **PixelSight is Authoritative**: The local LLM is an **interpretation layer only**. It does not perform image processing, and it must never generate, fabricate, overwrite, or substitute scientific measurements.
2. **Representation Guarantee**: PixelSight produces a 4× super-resolved representation approximately equivalent to 2.5 m spatial sampling (~2.5 m equivalent super-resolved representation). The Analyst never refers to outputs as "observed true 2.5 m satellite imagery".
3. **Reference-Aware Metrics**: When an external high-resolution reference is unavailable (`reference_available = false`), the Analyst explicitly notes that reference-dependent metrics (PSNR, SSIM, SAM) are withheld and cannot be interpreted as validated HR reconstruction accuracy.
4. **Uncertainty as Stochastic Dispersion**: Diffusion variance is explained as stochastic generative variability across DDPM reverse sampling paths, not converted into unsupported pixel correctness probabilities (e.g., never saying "there is a 92% chance this pixel is correct").
5. **No Medical/Yield Claims**: Crop analysis discusses NDVI canopy preservation and spectral fidelity; it never claims agronomic yield forecasts or disease pathology. Urban analysis discusses segmentation metrics (mIoU, Dice/F1) and notes that visual sharpness does not automatically equal superior segmentation accuracy. Disaster analysis distinguishes observed spectral surface differences from ground-truth verified physical disaster damage.

---

## Ollama Runtime & Local Processing

PixelSight Analyst operates **100% locally via Ollama**. It has **zero dependency on cloud LLMs** (no OpenAI, Anthropic, or Gemini API keys required). No satellite rasters or metadata leave your local workstation.

### Graceful Degradation
If Ollama is not installed, not running, or the configured model has not been pulled:
- The PixelSight application continues working completely normally.
- Scientific processing, super-resolution, metrics, reports, and UI remain 100% functional.
- The Analyst UI clearly indicates that Ollama is offline or the model is missing with setup instructions.

---

## Prerequisites & Installation

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/download):
- **Windows**: Install the Ollama Windows installer.
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **macOS**: Install Ollama macOS package.

### 2. Pull Recommended Local Model
We recommend a local 7B–8B instruct-class model:
```bash
ollama pull llama3.1:8b
```
Alternative supported models include `qwen2.5:7b`, `mistral:7b`, or `llama3.2:3b` for lightweight hardware.

### 3. Start the Ollama Service
Ensure the Ollama daemon is running:
```bash
ollama serve
```
(On Windows and macOS, opening the Ollama application starts the local service automatically on port 11434).

---

## Configuration

Configure the local analyst via environment variables or in `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `PIXELSIGHT_LLM_PROVIDER` | `ollama` | LLM backend runtime provider |
| `PIXELSIGHT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP API endpoint |
| `PIXELSIGHT_OLLAMA_MODEL` | `llama3.1:8b` | Configured instruct model tag |
| `PIXELSIGHT_OLLAMA_TIMEOUT` | `60.0` | Inference request timeout in seconds |

---

## API Endpoints

The local analyst exposes dedicated REST endpoints under `/api/v1/analyst`:

- `GET /api/v1/analyst/status`: Returns Ollama daemon reachability, model availability, and troubleshooting instructions.
- `GET /api/v1/analyst/models`: Lists installed local models in the Ollama runtime.
- `POST /api/v1/analyst/models/select`: Selects active local model at runtime.
- `POST /api/v1/analyst/jobs/{job_id}/explain`: Automatically builds scientific context from the job report and generates an authoritative interpretation.
- `POST /api/v1/analyst/jobs/{job_id}/chat`: Conducts bounded, grounded multi-turn conversation about the specific job.
- `POST /api/v1/analyst/batch/{batch_id}/summary`: Generates an analytical batch throughput and anomaly summary.
- `DELETE /api/v1/analyst/sessions/{session_id}`: Clears bounded conversation memory for a job or batch.

---

## Security & Prompt Injection Defense

1. **Data Isolation**: All satellite metadata, raster coordinates, filenames, and annotations are strictly treated as **DATA**.
2. **Input Sanitization**: Malicious prompt injections (e.g., attempts to override system guidelines, execute commands, or extract environment keys) are rejected deterministically before calling the LLM.
3. **Session Isolation**: Conversation memory is strictly isolated per `job_id`, preventing cross-job context contamination.
