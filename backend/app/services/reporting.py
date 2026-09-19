"""
PixelSight Multi-Format Research Reporting Service
====================================================
Generates JSON, Markdown, and standalone HTML reports
for each processing job across Research, Crop Monitoring,
Urban Analysis, and Disaster Management applications.
Provides full artifact manifests cataloguing every output produced during a run.

Scientific guarantees:
- Clearly distinguishes reference_available vs reference_unavailable evaluations.
- All metrics that require genuine HR reference are explicitly marked None when unavailable.
- Limitation statements are mandatory, not optional.
- Native-vs-SR consistency is strictly decoupled from ground-truth claims.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from textwrap import dedent
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(value: Any, fmt: str = ".4f") -> str:
    """Format numeric value or return em-dash."""
    if value is None:
        return "—"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _limitations_md(limitations: list[str]) -> str:
    return "\n".join(f"- {item}" for item in limitations)


def _metrics_table_md(evaluation: dict[str, Any]) -> str:
    if evaluation.get("status") != "reference_available":
        return "_No reference-dependent metrics available for this job._"
    rows = [
        f"| PSNR | {_safe(evaluation.get('psnr'), '.3f')} dB |",
        f"| SSIM | {_safe(evaluation.get('ssim'), '.4f')} |",
        f"| SAM  | {_safe(evaluation.get('sam'), '.3f')} ° |",
    ]
    header = "| Metric | Value |\n|--------|-------|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Build core report dictionary
# ---------------------------------------------------------------------------

def _build_report(
    *,
    job_id: str,
    input_metadata: dict[str, Any],
    preprocessing: list[dict[str, Any]],
    runtime_seconds: float,
    device: str,
    output_files: dict[str, str],
    status: str = "completed",
    evaluation: dict[str, Any] | None = None,
    urban_analysis: dict[str, Any] | None = None,
    uncertainty: dict[str, Any] | None = None,
    application: str = "research",
    application_data: dict[str, Any] | None = None,
    disaster_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = evaluation or {
        "status": "reference_unavailable",
        "psnr": None,
        "ssim": None,
        "sam": None,
        "reason": "No valid high-resolution reference was supplied.",
    }
    scientific_limitations = [
        "The SR product is a super-resolved representation, not observed 2.5m imagery.",
        "All pixel-level metrics are subject to resampling artefacts inherent to any bicubic interpolation chain.",
    ]
    if evaluation["status"] == "reference_available":
        scientific_limitations.append(
            "Reference-based metrics were computed against the uploaded input resampled to the SR grid "
            "(self-consistency diagnostic), NOT against an independent observed high-resolution reference."
        )
    else:
        scientific_limitations.append(
            "Reference-dependent metrics (PSNR, SSIM, SAM) are null because no valid "
            "high-resolution reference was supplied."
        )

    if application == "urban" or urban_analysis:
        urban_data = application_data if application == "urban" and application_data else (urban_analysis or {})
        scientific_limitations.extend(urban_data.get("limitations", []))
    elif application == "crop":
        crop_data = application_data or {}
        scientific_limitations.extend(crop_data.get("limitations", []))
    elif application == "disaster":
        disaster_data = application_data or {}
        scientific_limitations.extend(disaster_data.get("limitations", []))
    else:
        scientific_limitations.append(
            "Urban segmentation was not requested for this processing job."
        )

    if uncertainty and uncertainty.get("limitation"):
        scientific_limitations.append(uncertainty["limitation"])

    if evaluation.get("status") == "reference_available":
        evaluation_interpretation = (
            "The generated SR output was compared with the uploaded input after the input was resampled "
            "to the SR grid. "
            f"PSNR was {_safe(evaluation.get('psnr'), '.3f')} dB, "
            f"SSIM was {_safe(evaluation.get('ssim'), '.4f')}, and "
            f"SAM was {_safe(evaluation.get('sam'), '.3f')} degrees. "
            "These values describe input-to-output consistency; they are NOT evidence that the SR details "
            "are observed high-resolution ground truth."
        )
    else:
        evaluation_interpretation = (
            "A valid input-to-output evaluation could not be produced for this job. "
            f"{evaluation.get('reason', 'No evaluation values were returned.')}"
        )

    return {
        "job_id": job_id,
        "application": application,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": input_metadata,
        "preprocessing": {"operations": preprocessing},
        "super_resolution": {
            "model": "LDSR-S2",
            "scale": 4,
            "input_tile_size": [128, 128],
            "overlap": 12,
            "sampling_steps": 100,
            "description": "4x super-resolved representation (~2.5m equivalent)",
        },
        "runtime": {"device": device, "seconds": runtime_seconds},
        "outputs": output_files,
        "evaluation": evaluation,
        "evaluation_interpretation": evaluation_interpretation,
        "urban_analysis": urban_analysis or (application_data if application == "urban" else None),
        "crop_analysis": application_data if application == "crop" else None,
        "disaster_analysis": application_data if application == "disaster" else None,
        "disaster_metadata": disaster_metadata,
        "uncertainty": uncertainty,
        "scientific_limitations": scientific_limitations,
    }


# ---------------------------------------------------------------------------
# Format renderers
# ---------------------------------------------------------------------------

def _render_markdown(report: dict[str, Any]) -> str:
    job_id = report["job_id"]
    application = report.get("application", "research")
    created_at = report.get("created_at", "")
    evaluation = report.get("evaluation", {})
    uncertainty = report.get("uncertainty", {}) or {}
    urban = report.get("urban_analysis", {}) or {}
    crop = report.get("crop_analysis", {}) or {}
    disaster = report.get("disaster_analysis", {}) or {}
    runtime = report.get("runtime", {})
    sr = report.get("super_resolution", {})
    limitations = report.get("scientific_limitations", [])
    interpretation = report.get("evaluation_interpretation", "")

    app_title = {
        "research": "Core Engine & Research",
        "crop": "Crop Monitoring",
        "urban": "Urban Analysis",
        "disaster": "Disaster Management",
    }.get(application, application.capitalize())

    md = dedent(f"""\
    # PixelSight {app_title} Report — Job `{job_id}`

    **Application:** {app_title}
    **Status:** {report.get('status', 'unknown').upper()}
    **Generated:** {created_at}
    **Model:** {sr.get('model', 'LDSR-S2')} · Scale {sr.get('scale', 4)}x · {sr.get('sampling_steps', 100)} diffusion steps
    **Runtime:** {_safe(runtime.get('seconds'), '.2f')} s on `{runtime.get('device', 'unknown')}`

    ---

    ## 1. Input Metadata

    | Field | Value |
    |-------|-------|
    | Width | {report.get('input', {}).get('width', '—')} px |
    | Height | {report.get('input', {}).get('height', '—')} px |
    | Bands | {', '.join(report.get('input', {}).get('band_names', []) or [])} |
    | CRS | {report.get('input', {}).get('crs', '—')} |
    | Compatible | {report.get('input', {}).get('compatible', '—')} |

    ---

    ## 2. Image Quality & Consistency Metrics

    {_metrics_table_md(evaluation)}

    **Interpretation:**
    {interpretation}

    ---

    ## 3. Uncertainty Analysis

    | Metric | Value |
    |--------|-------|
    | Mean uncertainty | {_safe(uncertainty.get('mean'))} |
    | Peak uncertainty | {_safe(uncertainty.get('peak'))} |
    | Low-confidence pixels | {_safe(uncertainty.get('low_percent'), '.1f')} % |
    | High-risk pixels | {_safe(uncertainty.get('high_percent'), '.1f')} % |

    *{uncertainty.get('limitation', 'Uncertainty reflects stochastic diffusion variation across random seeds.')}*
    """)

    # Section 4: Application-specific content
    if application == "crop" and crop:
        stats = crop.get("statistics", {})
        nat = stats.get("native", {})
        sr_st = stats.get("super_resolution", {})
        cm = crop.get("consistency_metrics", {})
        md += dedent(f"""
        ---

        ## 4. Crop Vegetation Analysis (NDVI)

        **Scientific Status:** {crop.get('scientific_status', 'Native-vs-SR consistency analysis')}
        **Formula:** `{crop.get('formula', '(B08 - B04)/(B08 + B04)')}`

        | Metric | Native (10 m) | Super-Resolved (4x, ~2.5 m) |
        |--------|---------------|-----------------------------|
        | Mean NDVI | {_safe(nat.get('mean'), '.4f')} | {_safe(sr_st.get('mean'), '.4f')} |
        | Std Dev   | {_safe(nat.get('std'), '.4f')}  | {_safe(sr_st.get('std'), '.4f')}  |
        | MAE (Native vs SR) | — | {_safe(cm.get('mae'), '.4f')} |
        | RMSE (Native vs SR) | — | {_safe(cm.get('rmse'), '.4f')} |

        **Interpretations:**
        {chr(10).join('- ' + item for item in crop.get('interpretations', []))}
        """)
    elif application == "disaster" and disaster:
        d_stats = disaster.get("statistics", {})
        rel = d_stats.get("reliability_breakdown", {})
        md += dedent(f"""
        ---

        ## 4. Disaster Temporal Change Detection

        **Scientific Status:** {disaster.get('scientific_status', 'Research analysis — no independent ground truth')}
        **Methodology:** {disaster.get('methodology', 'Spectral difference magnitude')}

        | Metric | Value |
        |--------|-------|
        | Changed Area (pixels) | {d_stats.get('changed_pixels', '—')} |
        | Change Area Percentage | {_safe(d_stats.get('change_percentage'), '.2f')} % |
        | Applied Change Threshold | {_safe(d_stats.get('change_threshold_applied'), '.3f')} |
        | Lower-Uncertainty Fraction | {_safe(rel.get('lower_uncertainty_fraction_of_change', 0) * 100, '.1f')} % |
        | Moderate-Uncertainty Fraction | {_safe(rel.get('moderate_uncertainty_fraction_of_change', 0) * 100, '.1f')} % |
        | High-Uncertainty Fraction | {_safe(rel.get('high_uncertainty_fraction_of_change', 0) * 100, '.1f')} % |

        **Interpretations:**
        {chr(10).join('- ' + item for item in disaster.get('interpretations', []))}
        """)
    elif urban:
        md += dedent(f"""
        ---

        ## 4. Urban Land-Cover Classification

        | Class | Objects | Area |
        |-------|---------|------|
        | Trees/Vegetation | {urban.get('trees', '—')} | — |
        | Houses/Built-up | {urban.get('houses', '—')} | — |
        | Other | {urban.get('other_objects', '—')} | — |

        {urban.get('interpretation', '')}
        """)

    md += dedent(f"""
    ---

    ## 5. Scientific Limitations

    {_limitations_md(limitations)}

    ---

    *This report was generated by PixelSight — Deep Learning Super-Resolution Mapping (SRM).*
    *SIH 2026 Problem Statement SIH26142: "Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imageries"*
    """)
    return md


def _render_html(report: dict[str, Any], markdown_content: str) -> str:
    job_id = report["job_id"]
    status_color = "#22c55e" if report.get("status") == "completed" else "#f59e0b"

    def md_to_html(md: str) -> str:
        lines = md.split("\n")
        html_lines = []
        in_table = False
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                html_lines.append("<pre><code>" if in_code else "</code></pre>")
                continue
            if in_code:
                html_lines.append(f"<code>{line}</code>")
                continue
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("| "):
                if not in_table:
                    in_table = True
                    html_lines.append('<table class="data-table"><tbody>')
                cells = [c.strip() for c in line.strip("|").split("|")]
                if any(c.startswith("---") for c in cells):
                    continue
                row = "".join(f"<td>{c}</td>" for c in cells)
                html_lines.append(f"<tr>{row}</tr>")
            else:
                if in_table:
                    in_table = False
                    html_lines.append("</tbody></table>")
                if line.startswith("- "):
                    html_lines.append(f"<li>{line[2:]}</li>")
                elif line.startswith("---"):
                    html_lines.append("<hr/>")
                elif line.strip():
                    styled = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                    styled = re.sub(r"\*(.+?)\*", r"<em>\1</em>", styled)
                    styled = re.sub(r"`(.+?)`", r"<code>\1</code>", styled)
                    html_lines.append(f"<p>{styled}</p>")
        if in_table:
            html_lines.append("</tbody></table>")
        return "\n".join(html_lines)

    body = md_to_html(markdown_content)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PixelSight Report — {job_id}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #e2e8f0;
    background: #0f172a;
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }}
  h1 {{ color: #38bdf8; border-bottom: 2px solid #1e293b; padding-bottom: 0.5rem; }}
  h2 {{ color: #7dd3fc; margin-top: 2rem; }}
  h3 {{ color: #bae6fd; }}
  hr {{ border: none; border-top: 1px solid #1e293b; margin: 2rem 0; }}
  code {{ background: #1e293b; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }}
  pre {{ background: #1e293b; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
  .data-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  .data-table td {{ padding: 0.5rem 0.75rem; border: 1px solid #334155; font-size: 0.92rem; }}
  .data-table tr:first-child td {{ font-weight: 600; background: #1e293b; }}
  li {{ margin-bottom: 0.4rem; }}
  .badge {{ display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px;
           background: {status_color}22; color: {status_color}; font-weight: 600; font-size: 0.8rem; }}
</style>
</head>
<body>
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
    <span class="badge">{report.get('status', 'COMPLETED').upper()}</span>
    <span style="color:#64748b; font-size:0.85rem;">Job: {job_id}</span>
  </div>
  {body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public service interface
# ---------------------------------------------------------------------------

def write_report(
    path: str | Path,
    *,
    job_id: str,
    input_metadata: dict[str, Any],
    preprocessing: list[dict[str, Any]],
    runtime_seconds: float,
    device: str,
    output_files: dict[str, str],
    status: str = "completed",
    evaluation: dict[str, Any] | None = None,
    urban_analysis: dict[str, Any] | None = None,
    uncertainty: dict[str, Any] | None = None,
    application: str = "research",
    application_data: dict[str, Any] | None = None,
    disaster_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write report.json, report.md, and report.html for a processing job."""
    report = _build_report(
        job_id=job_id,
        input_metadata=input_metadata,
        preprocessing=preprocessing,
        runtime_seconds=runtime_seconds,
        device=device,
        output_files=output_files,
        status=status,
        evaluation=evaluation,
        urban_analysis=urban_analysis,
        uncertainty=uncertainty,
        application=application,
        application_data=application_data,
        disaster_metadata=disaster_metadata,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 2. Markdown
    md_path = path.with_suffix(".md")
    md_content = _render_markdown(report)
    md_path.write_text(md_content, encoding="utf-8")

    # 3. HTML
    html_path = path.with_suffix(".html")
    html_content = _render_html(report, md_content)
    html_path.write_text(html_content, encoding="utf-8")

    return report


def write_manifest(
    job_dir: Path,
    job_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Write results/{job_id}/manifest.json cataloguing all artifacts.

    The manifest provides a single-file index of every generated artifact
    so the frontend and downstream tools can discover outputs without
    hardcoded path assumptions.
    """
    artifact_entries: list[dict[str, Any]] = []

    def _try_add(rel_path: str, artifact_type: str, description: str) -> None:
        full = job_dir / rel_path
        if full.exists():
            artifact_entries.append({
                "path": rel_path,
                "type": artifact_type,
                "description": description,
                "size_bytes": full.stat().st_size,
            })

    # Core inputs & preprocessing
    _try_add("preprocessing/normalized.tif", "geotiff", "Preprocessed float32 4-band normalized input")
    _try_add("input/original_preview.png", "preview_png", "RGB preview of original input")

    # Super-resolution
    _try_add("super_resolution/sr.tif", "geotiff", "LDSR-S2 4x super-resolved GeoTIFF (~2.5 m equivalent)")
    _try_add("super_resolution/sr_preview.png", "preview_png", "RGB preview of super-resolved output")

    # Uncertainty
    _try_add("uncertainty/uncertainty_map.tif", "geotiff", "Stochastic diffusion uncertainty map (GeoTIFF)")
    _try_add("uncertainty/uncertainty_map.png", "preview_png", "Uncertainty map visualization (PNG)")

    # Legacy urban analysis paths
    _try_add("analysis/urban_planning_map.png", "preview_png", "Urban/vegetation classification map")
    _try_add("analysis/urban_classes.tif", "geotiff", "Urban classification raster (GeoTIFF)")
    _try_add("analysis/input_urban_planning_map.png", "preview_png", "Input-resolution classification map")
    _try_add("analysis/input_urban_classes.tif", "geotiff", "Input-resolution classification raster (GeoTIFF)")

    # Dedicated Crop Monitoring outputs
    _try_add("application/crop/ndvi_native.tif", "geotiff", "Native resolution NDVI raster")
    _try_add("application/crop/ndvi_sr.tif", "geotiff", "4x super-resolved NDVI raster (~2.5 m equivalent)")
    _try_add("application/crop/ndvi_difference.tif", "geotiff", "NDVI difference map (SR - Native)")
    _try_add("application/crop/previews/ndvi_native.png", "preview_png", "Native NDVI color preview")
    _try_add("application/crop/previews/ndvi_sr.png", "preview_png", "Super-resolved NDVI color preview")
    _try_add("application/crop/previews/ndvi_difference.png", "preview_png", "NDVI difference preview")
    _try_add("application/crop/metrics.json", "metrics_json", "Crop vegetation analysis statistics & consistency metrics")

    # Dedicated Urban Analysis outputs
    _try_add("application/urban/segmentation_native.tif", "geotiff", "Native input segmentation raster")
    _try_add("application/urban/segmentation_sr.tif", "geotiff", "4x super-resolved segmentation raster")
    _try_add("application/urban/builtup.tif", "geotiff", "Built-up area mask raster")
    _try_add("application/urban/urban_difference.tif", "geotiff", "Urban land-cover difference map")
    _try_add("application/urban/previews/segmentation_native.png", "preview_png", "Native segmentation preview")
    _try_add("application/urban/previews/segmentation_sr.png", "preview_png", "Super-resolved segmentation preview")
    _try_add("application/urban/previews/builtup_preview.png", "preview_png", "Built-up mask preview")
    _try_add("application/urban/previews/urban_difference.png", "preview_png", "Segmentation difference preview")
    _try_add("application/urban/metrics.json", "metrics_json", "Urban indicators and class distributions")

    # Dedicated Disaster Management outputs
    _try_add("application/disaster/pre_event/sr_pre.tif", "geotiff", "Pre-event 4x super-resolved GeoTIFF")
    _try_add("application/disaster/post_event/sr_post.tif", "geotiff", "Post-event 4x super-resolved GeoTIFF")
    _try_add("application/disaster/change_map.tif", "geotiff", "Multi-spectral change magnitude raster")
    _try_add("application/disaster/affected_area.tif", "geotiff", "Potential affected region mask")
    _try_add("application/disaster/uncertainty.tif", "geotiff", "Combined super-resolution uncertainty raster")
    _try_add("application/disaster/previews/change_preview.png", "preview_png", "Change magnitude preview")
    _try_add("application/disaster/previews/affected_area_preview.png", "preview_png", "Uncertainty-aware affected area preview")
    _try_add("application/disaster/previews/pre_sr_preview.png", "preview_png", "Pre-event SR RGB preview")
    _try_add("application/disaster/previews/post_sr_preview.png", "preview_png", "Post-event SR RGB preview")
    _try_add("application/disaster/metrics.json", "metrics_json", "Disaster change and reliability metrics")

    # Reports
    _try_add("report/report.json", "report_json", "Primary structured evaluation report (JSON)")
    _try_add("report/report.md", "report_markdown", "Human-readable evaluation report (Markdown)")
    _try_add("report/report.html", "report_html", "Standalone HTML evaluation report")

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "application": report.get("application", "research"),
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": report.get("status", "completed"),
        "model": "LDSR-S2",
        "scale_factor": 4,
        "artifacts": artifact_entries,
        "evaluation_summary": {
            "reference_type": report.get("evaluation", {}).get("status", "reference_unavailable"),
            "psnr": report.get("evaluation", {}).get("psnr"),
            "ssim": report.get("evaluation", {}).get("ssim"),
            "sam": report.get("evaluation", {}).get("sam"),
        },
        "scientific_limitations": report.get("scientific_limitations", []),
    }

    if report.get("disaster_metadata"):
        manifest["disaster_metadata"] = report["disaster_metadata"]

    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
