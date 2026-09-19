import React, { useState } from 'react';
import {
  Download, FileText, Image as ImageIcon, RotateCcw,
  ShieldAlert, CheckCircle2, Trees, Building2,
  BarChart3, Globe2, Layers, Activity, Cpu, BookOpen,
  ChevronRight, AlertTriangle, Info, ExternalLink,
  Thermometer, Radar, Map,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import { resultFileUrl } from '../api/srmApi.js';

// ─── Utility components ────────────────────────────────────────────────────

function Metric({ label, value, note, highlight }) {
  return (
    <div className="stat-card">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-2xl font-bold font-mono mt-2 ${highlight ? 'text-amber-300' : 'text-cyan-300'}`}>
        {value ?? '—'}
      </div>
      {note && <div className="text-xs text-slate-600 mt-1">{note}</div>}
    </div>
  );
}

function MetricRow({ label, value, suffix = '', note }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 border-b border-white/[0.05] last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <div className="text-right">
        <span className="font-mono text-slate-200 text-sm">{value ?? '—'}{suffix}</span>
        {note && <div className="text-xs text-slate-600">{note}</div>}
      </div>
    </div>
  );
}

function EvaluationBar({ label, value, display, scale, color, note }) {
  const numericValue = Number(value);
  const width = Number.isFinite(numericValue) ? Math.max(3, Math.min(100, (numericValue / scale) * 100)) : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">{display}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${width}%`, background: color }} />
      </div>
      <div className="text-[10px] text-slate-600">{note}</div>
    </div>
  );
}

function ComparisonBar({ value, maximum, color }) {
  const width = maximum > 0 ? Math.max(4, (value / maximum) * 100) : 0;
  return (
    <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${width}%`, background: color }} />
    </div>
  );
}

function SciNote({ children, type = 'info' }) {
  const styles = {
    info: { border: 'border-blue-500/20', bg: 'bg-blue-500/[0.04]', text: 'text-blue-200/80', icon: <Info className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" /> },
    warning: { border: 'border-amber-500/20', bg: 'bg-amber-500/[0.04]', text: 'text-amber-200/80', icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" /> },
    success: { border: 'border-emerald-500/20', bg: 'bg-emerald-500/[0.04]', text: 'text-emerald-200/80', icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" /> },
  };
  const s = styles[type] || styles.info;
  return (
    <div className={`flex items-start gap-2 rounded-lg border ${s.border} ${s.bg} p-3 text-xs ${s.text} leading-relaxed`}>
      {s.icon}<span>{children}</span>
    </div>
  );
}

function TabButton({ id, label, icon: Icon, active, onClick }) {
  return (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap
        ${active
          ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
          : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.03]'
        }`}
    >
      {Icon && <Icon className="w-3.5 h-3.5" />}
      {label}
    </button>
  );
}

function SectionHeader({ title, subtitle, badge }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-3 mb-1">
        <h2 className="text-xl font-bold text-white">{title}</h2>
        {badge && (
          <span className="text-xs font-mono px-2 py-0.5 rounded-full border border-cyan-500/30 text-cyan-400 bg-cyan-500/10">
            {badge}
          </span>
        )}
      </div>
      {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
    </div>
  );
}

// ─── Tab content panels ────────────────────────────────────────────────────

function OverviewTab({ job, report, outputs, file }) {
  const evaluation = report.evaluation || {};
  const hasEval = evaluation.status === 'reference_available' &&
    [evaluation.psnr, evaluation.ssim, evaluation.sam].every(v => Number.isFinite(Number(v)));
  const runtime = report.runtime || {};
  const sr = report.super_resolution || {};
  const input = report.input || {};

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Job Overview"
        subtitle={`Job ${job.job_id} — LDSR-S2 4× Super-Resolution`}
        badge="completed"
      />

      {/* Key stats */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Metric label="Model" value="LDSR-S2" note="Latent Diffusion SR" />
        <Metric label="Scale" value={`${sr.scale ?? 4}×`} note="128 → 512 px tiles" />
        <Metric label="Diffusion steps" value={sr.sampling_steps ?? 100} note="per tile" />
        <Metric label="Runtime" value={`${Number(runtime.seconds || 0).toFixed(1)} s`} note={runtime.device || 'device'} />
      </div>

      {/* Input details */}
      <div className="card p-6">
        <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold mb-4">Input Raster</div>
        <div className="grid sm:grid-cols-2 gap-x-8">
          <MetricRow label="Width" value={input.width} suffix=" px" />
          <MetricRow label="Height" value={input.height} suffix=" px" />
          <MetricRow label="Bands" value={input.band_names?.join(' / ') || input.bands} />
          <MetricRow label="CRS" value={input.crs} />
          <MetricRow label="Format" value={input.format} />
          <MetricRow label="Compatible" value={input.compatible ? 'Yes' : 'No'} />
        </div>
      </div>

      {/* Evaluation summary */}
      <div className="card p-6">
        <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold mb-4">Evaluation Summary</div>
        {hasEval ? (
          <div className="space-y-4">
            <div className="grid sm:grid-cols-3 gap-3">
              <Metric label="PSNR" value={`${Number(evaluation.psnr).toFixed(2)} dB`} note="input vs SR" />
              <Metric label="SSIM" value={Number(evaluation.ssim).toFixed(4)} note="input vs SR" />
              <Metric label="SAM" value={`${Number(evaluation.sam).toFixed(2)}°`} note="input vs SR" />
            </div>
            <SciNote type="warning">
              These metrics compare the uploaded input (resampled to the SR grid) against the generated SR output.
              They are <strong>self-consistency diagnostics</strong>, NOT validation against independent high-resolution ground truth.
            </SciNote>
          </div>
        ) : (
          <SciNote type="info">
            {evaluation.reason || 'Reference-based metrics are not available. Process a new image with a high-resolution reference to generate evaluation metrics.'}
          </SciNote>
        )}
      </div>

      {/* Download shortcuts */}
      <div className="grid sm:grid-cols-3 gap-3">
        <a className="btn-ghost justify-center" href={file('super_resolution/sr.tif')} download>
          <ImageIcon className="w-4 h-4" /> SR GeoTIFF
        </a>
        <a className="btn-ghost justify-center" href={file('report/report.html')} target="_blank" rel="noreferrer">
          <ExternalLink className="w-4 h-4" /> HTML Report
        </a>
        <a className="btn-ghost justify-center" href={file('report/report.json')} download="pixelsight-report.json">
          <Download className="w-4 h-4" /> JSON Report
        </a>
      </div>
    </div>
  );
}

function SuperResolutionTab({ file, outputs }) {
  const original = file(outputs.original_preview || 'input/original_preview.png');
  const sr = file(outputs.super_resolution_preview || 'super_resolution/sr_preview.png');
  const srTif = file(outputs.super_resolution || 'super_resolution/sr.tif');

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Super-Resolution Output"
        subtitle="Drag the slider to compare the original input and the 4× LDSR-S2 output."
        badge="LDSR-S2"
      />

      <div className="card overflow-hidden">
        <CompareSlider beforeSrc={original} afterSrc={sr} />
        <div className="px-5 py-3 text-xs text-slate-500 border-t border-white/[0.05] flex justify-between">
          <span>Input: 10 m native resolution</span>
          <span>Output: ~2.5 m equivalent (4×)</span>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Input Preview</div>
          <img src={original} alt="Original input preview" className="w-full rounded-lg object-cover" />
          <a href={original} download className="btn-ghost mt-3 w-full justify-center text-xs">
            <Download className="w-3.5 h-3.5" /> Download preview
          </a>
        </div>
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-cyan-400 mb-3">SR Output Preview</div>
          <img src={sr} alt="Super-resolved output preview" className="w-full rounded-lg object-cover" />
          <a href={sr} download className="btn-ghost mt-3 w-full justify-center text-xs">
            <Download className="w-3.5 h-3.5" /> Download preview
          </a>
        </div>
      </div>

      <div className="card p-5">
        <a className="btn-primary w-full justify-center" href={srTif} download>
          <ImageIcon className="w-4 h-4" /> Download Super-Resolved GeoTIFF
        </a>
        <p className="text-xs text-slate-600 mt-3 text-center">
          GeoTIFF preserves the original georeferencing metadata at 4× resolution.
        </p>
      </div>

      <SciNote type="warning">
        The output is a super-resolved representation generated by a stochastic diffusion model. Sub-pixel detail
        may reflect model priors, not observed ground truth. Always validate with independent high-resolution references.
      </SciNote>
    </div>
  );
}

function UncertaintyTab({ file, report, outputs }) {
  const uncertainty = report.uncertainty || outputs.uncertainty || {};
  const uncertaintyMap = file(uncertainty.map || outputs.uncertainty_map || 'uncertainty/uncertainty_map.png');
  const uncertaintyTif = file(outputs.uncertainty_geotiff || 'uncertainty/uncertainty_map.tif');

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Uncertainty Analysis"
        subtitle="Stochastic diffusion variation — pixel-wise std across N independent LDSR-S2 seeds."
        badge="NOT Bayesian"
      />

      <div className="grid md:grid-cols-[1fr_0.8fr] gap-6 items-start">
        <div>
          <div className="card overflow-hidden border border-amber-500/20">
            <img src={uncertaintyMap} alt="SR uncertainty map" className="w-full object-contain" />
            <div className="px-4 py-2 text-xs text-amber-300/70 border-t border-amber-500/20">
              Bright = high reconstruction disagreement · Dark = low uncertainty
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-3">
            <a href={uncertaintyMap} download className="btn-ghost text-xs justify-center">
              <Download className="w-3.5 h-3.5" /> PNG Map
            </a>
            <a href={uncertaintyTif} download className="btn-ghost text-xs justify-center">
              <Download className="w-3.5 h-3.5" /> GeoTIFF
            </a>
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Metric label="Mean σ" value={uncertainty.mean != null ? uncertainty.mean.toFixed(4) : null} note="normalized std" />
            <Metric label="Peak σ" value={uncertainty.peak != null ? uncertainty.peak.toFixed(4) : null} note="maximum value" highlight />
            <Metric label="Low confidence" value={uncertainty.low_percent != null ? `${uncertainty.low_percent.toFixed(1)}%` : null} note="pixels" />
            <Metric label="High risk" value={uncertainty.high_percent != null ? `${uncertainty.high_percent.toFixed(1)}%` : null} note="pixels" highlight />
          </div>
        </div>
      </div>

      <div className="card p-5 space-y-3">
        <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Mechanism</div>
        <p className="text-sm text-slate-400 leading-relaxed">
          Uncertainty is computed as the pixel-wise standard deviation across multiple independent LDSR-S2
          diffusion runs using different random seeds. This quantifies <strong className="text-slate-200">stochastic generative variability</strong>,
          not Bayesian epistemic model uncertainty.
        </p>
        <SciNote type="warning">
          {uncertainty.limitation ||
            'High uncertainty does not prove that SR detail is incorrect — it indicates that the diffusion model is sensitive to the noise schedule in that region. Regions with high uncertainty should be verified against reference imagery.'}
        </SciNote>
      </div>
    </div>
  );
}

function SegmentationTab({ file, report, outputs }) {
  const urban = report.urban_analysis || outputs.urban_analysis || {};
  const planningMap = file(urban.map || outputs.urban_planning_map || 'analysis/urban_planning_map.png');
  const classifiedRaster = file(urban.classified_raster || outputs.urban_classification || 'analysis/urban_classes.tif');
  const inputPlanningMap = file(outputs.input_urban_planning_map || 'analysis/input_urban_planning_map.png');
  const comparison = urban.comparison || {};
  const comparisonRows = Object.entries(comparison.classes || {});
  const maximumCount = Math.max(1, ...comparisonRows.flatMap(([, item]) => [item.input_object_count || 0, item.sr_object_count || 0]));

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Land Cover Classification"
        subtitle="WorldCover-proxy segmentation applied to the SR output. Objects are connected-region estimates."
        badge="proxy labels"
      />

      <div className="grid sm:grid-cols-3 gap-3">
        <Metric label="Vegetation regions" value={urban.trees ?? urban.tree_clusters} note="connected regions" />
        <Metric label="Built-up regions" value={urban.houses ?? urban.estimated_building_clusters} note="connected regions" />
        <Metric label="Other classes" value={urban.other_objects} note="all remaining" />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">SR Output Classification</div>
          <img src={planningMap} alt="Urban planning map" className="w-full rounded-lg" />
          <div className="grid grid-cols-2 gap-2 mt-3">
            <a href={planningMap} download className="btn-ghost text-xs justify-center">
              <Download className="w-3.5 h-3.5" /> PNG
            </a>
            <a href={classifiedRaster} download className="btn-ghost text-xs justify-center">
              <Download className="w-3.5 h-3.5" /> GeoTIFF
            </a>
          </div>
        </div>
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Input Classification</div>
          <img src={inputPlanningMap} alt="Input classification map" className="w-full rounded-lg" />
          <p className="text-xs text-slate-600 mt-2">Segmentation applied directly to the native 10 m input for comparison.</p>
        </div>
      </div>

      {/* Object class table */}
      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-4">Class-wise Counts</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-slate-500 border-b border-white/[0.08]">
                <th className="py-2 pr-4">Class</th>
                <th className="py-2 pr-4">Objects</th>
                <th className="py-2 pr-4">Pixels</th>
                <th className="py-2">Area %</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(urban.object_counts || {}).map(([name, item]) => (
                <tr key={name} className="border-b border-white/[0.05] text-slate-300">
                  <td className="py-2 pr-4">{item.label || name}</td>
                  <td className="py-2 pr-4 font-mono text-cyan-300">{item.object_count ?? '—'}</td>
                  <td className="py-2 pr-4 font-mono">{item.pixel_count ?? '—'}</td>
                  <td className="py-2 font-mono">{item.area_percent != null ? `${item.area_percent.toFixed(1)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison bars */}
      {comparisonRows.length > 0 && (
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-4">Input vs SR Object Counts</div>
          <div className="space-y-5">
            {comparisonRows.map(([name, item]) => (
              <div key={name}>
                <div className="flex items-center justify-between gap-4 mb-2">
                  <span className="text-sm font-medium text-slate-200">{item.label || name}</span>
                  <span className={`text-xs font-mono ${(item.object_count_change ?? 0) >= 0 ? 'text-emerald-300' : 'text-amber-300'}`}>
                    {item.object_count_change >= 0 ? '+' : ''}{item.object_count_change} objects
                  </span>
                </div>
                <div className="grid grid-cols-[52px_1fr_36px] items-center gap-3 mb-1">
                  <span className="text-[10px] uppercase text-slate-500">Input</span>
                  <ComparisonBar value={item.input_object_count || 0} maximum={maximumCount} color="#14b8a6" />
                  <span className="text-right text-xs font-mono text-teal-300">{item.input_object_count ?? 0}</span>
                </div>
                <div className="grid grid-cols-[52px_1fr_36px] items-center gap-3">
                  <span className="text-[10px] uppercase text-cyan-400">SR</span>
                  <ComparisonBar value={item.sr_object_count || 0} maximum={maximumCount} color="#22d3ee" />
                  <span className="text-right text-xs font-mono text-cyan-300">{item.sr_object_count ?? 0}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Interpretation */}
      {urban.interpretation && (
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-3">Interpretation</div>
          <p className="text-sm text-slate-300 leading-relaxed">{urban.interpretation}</p>
        </div>
      )}

      <SciNote type="warning">
        Segmentation uses proxy land cover labels (WorldCover-inspired). Object counts are connected-region estimates
        and are NOT cadastral records, property surveys, or aerial photogrammetry results.
        {urban.limitations?.[0] && ` ${urban.limitations[0]}`}
      </SciNote>
    </div>
  );
}

function EvaluationTab({ report }) {
  const evaluation = report.evaluation || {};
  const hasEval = evaluation.status === 'reference_available' &&
    [evaluation.psnr, evaluation.ssim, evaluation.sam].every(v => Number.isFinite(Number(v)));
  const interpretation = report.evaluation_interpretation || '';

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Image Quality Metrics"
        subtitle="PSNR, SSIM, and SAM comparing the input (resampled) against the SR output."
        badge={hasEval ? 'reference_available' : 'reference_unavailable'}
      />

      {hasEval ? (
        <>
          <div className="grid sm:grid-cols-3 gap-4">
            <Metric label="PSNR" value={`${Number(evaluation.psnr).toFixed(3)} dB`} note="input vs SR output" />
            <Metric label="SSIM" value={Number(evaluation.ssim).toFixed(4)} note="structural similarity" />
            <Metric label="SAM" value={`${Number(evaluation.sam).toFixed(3)}°`} note="spectral angle mapper" highlight />
          </div>

          <div className="card p-5 space-y-5">
            <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Metric Visualization</div>
            <EvaluationBar
              label="PSNR"
              value={evaluation.psnr}
              display={`${Number(evaluation.psnr).toFixed(3)} dB`}
              scale={50}
              color="#22d3ee"
              note="Visual scale capped at 50 dB. Higher is generally better for self-consistency."
            />
            <EvaluationBar
              label="SSIM"
              value={evaluation.ssim}
              display={Number(evaluation.ssim).toFixed(4)}
              scale={1}
              color="#34d399"
              note="Scale 0–1. Higher = greater structural similarity between input and SR output."
            />
            <EvaluationBar
              label="SAM (inverted)"
              value={Math.max(0, 180 - Number(evaluation.sam))}
              display={`${Number(evaluation.sam).toFixed(3)}°`}
              scale={180}
              color="#f59e0b"
              note="Lower angular error is better. Bar shows 180° − SAM for visual clarity."
            />
          </div>
        </>
      ) : (
        <div className="card p-6">
          <SciNote type="info">
            {evaluation.reason || 'No high-resolution reference was available. Upload a reference image to compute ground-truth metrics (PSNR, SSIM, SAM).'}
          </SciNote>
        </div>
      )}

      <div className="card p-5 space-y-3">
        <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Interpretation</div>
        <p className="text-sm text-slate-400 leading-relaxed">{interpretation || 'No evaluation interpretation was generated for this job.'}</p>
        <SciNote type="warning">
          These metrics compare the uploaded input with the super-resolved output. This is a self-consistency diagnostic —
          it does NOT constitute validation against an independent observed high-resolution reference.
        </SciNote>
      </div>
    </div>
  );
}

function ScientificStatusTab({ report }) {
  const limitations = report.scientific_limitations || [];
  const sr = report.super_resolution || {};
  const runtime = report.runtime || {};

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Scientific Status & Limitations"
        subtitle="Transparency notice — all methodological constraints are listed here."
        badge="research integrity"
      />

      {/* Model details */}
      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold mb-4">Model & Protocol</div>
        <div className="grid sm:grid-cols-2 gap-x-8">
          <MetricRow label="Model" value={sr.model || 'LDSR-S2'} />
          <MetricRow label="Scale factor" value={`${sr.scale || 4}×`} />
          <MetricRow label="Diffusion steps" value={sr.sampling_steps || 100} />
          <MetricRow label="Tile size" value={sr.input_tile_size ? sr.input_tile_size.join(' × ') : '128 × 128'} suffix=" px" />
          <MetricRow label="Overlap" value={sr.overlap ?? 12} suffix=" px" />
          <MetricRow label="Device" value={runtime.device || 'cpu'} />
          <MetricRow label="Runtime" value={runtime.seconds ? Number(runtime.seconds).toFixed(2) : '—'} suffix=" s" />
          <MetricRow label="Output GSD" value="~2.5 m equivalent" note="not physical observation" />
        </div>
      </div>

      {/* Limitations */}
      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-amber-400 font-semibold mb-4">Scientific Limitations</div>
        <div className="space-y-2">
          {limitations.length === 0 ? (
            <p className="text-sm text-slate-500">No limitations recorded for this job.</p>
          ) : (
            limitations.map((item, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-slate-400 leading-relaxed py-1.5 border-b border-white/[0.04] last:border-0">
                <AlertTriangle className="w-3 h-3 text-amber-400/70 flex-shrink-0 mt-0.5" />
                <span>{item}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* SIH Context */}
      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-3">Research Context</div>
        <p className="text-sm text-slate-400 leading-relaxed">
          PixelSight implements <strong className="text-slate-200">SIH 2026 Problem Statement SIH26142</strong>:{' '}
          "Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries."
          The system uses Sentinel-2 compatible 4-band (B02/B03/B04/B08) imagery as input.
        </p>
        <div className="mt-3 grid sm:grid-cols-2 gap-3 text-xs text-slate-500">
          <div>• LDSR-S2: Latent Diffusion SR for Sentinel-2 MSI</div>
          <div>• SRM: Sub-pixel land-cover mapping at 2.5 m</div>
          <div>• Uncertainty: Stochastic diffusion variability</div>
          <div>• Downstream: WorldCover-proxy classification</div>
        </div>
      </div>
    </div>
  );
}

function DownloadsTab({ file, report, outputs, job }) {
  const reportUrl = file(outputs.report || 'report/report.json');
  const htmlReportUrl = file(outputs.report_html || 'report/report.html');
  const mdReportUrl = file(outputs.report_markdown || 'report/report.md');
  const manifestUrl = file(outputs.manifest || 'manifest.json');
  const srTif = file(report.outputs?.super_resolution || 'super_resolution/sr.tif');
  const classifiedRaster = file(outputs.urban_classification || 'analysis/urban_classes.tif');
  const uncertaintyTif = file(outputs.uncertainty_geotiff || 'uncertainty/uncertainty_map.tif');
  const original = file(outputs.original_preview || 'input/original_preview.png');

  const DownloadRow = ({ href, filename, label, desc, ext }) => (
    <div className="flex items-center justify-between gap-4 py-3 border-b border-white/[0.05] last:border-0">
      <div>
        <div className="text-sm font-medium text-slate-200">{label}</div>
        <div className="text-xs text-slate-500 mt-0.5">{desc}</div>
      </div>
      <a href={href} download={filename} className="btn-ghost text-xs whitespace-nowrap">
        <Download className="w-3.5 h-3.5" /> {ext}
      </a>
    </div>
  );

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Downloads & Reports"
        subtitle="All artifacts generated during this processing job."
        badge={`job ${job.job_id}`}
      />

      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold mb-4">Raster Outputs</div>
        <DownloadRow href={srTif} filename="pixelsight-sr.tif" label="Super-Resolved GeoTIFF" desc="LDSR-S2 4× output at ~2.5 m equivalent resolution" ext=".tif" />
        <DownloadRow href={uncertaintyTif} filename="uncertainty-map.tif" label="Uncertainty Map GeoTIFF" desc="Stochastic diffusion std (uncertainty) per pixel" ext=".tif" />
        <DownloadRow href={classifiedRaster} filename="urban-classes.tif" label="Classification Raster" desc="WorldCover-proxy land cover classes (GeoTIFF)" ext=".tif" />
        <DownloadRow href={original} filename="original-preview.png" label="Original Preview PNG" desc="RGB preview of the uploaded input raster" ext=".png" />
      </div>

      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-emerald-400 font-semibold mb-4">Reports</div>
        <DownloadRow href={reportUrl} filename="pixelsight-report.json" label="JSON Report" desc="Machine-readable structured evaluation report" ext=".json" />
        <DownloadRow href={mdReportUrl} filename="pixelsight-report.md" label="Markdown Report" desc="Human-readable GitHub-flavoured Markdown report" ext=".md" />
        <DownloadRow
          href={htmlReportUrl}
          filename="pixelsight-report.html"
          label="HTML Report"
          desc="Standalone interactive HTML report (no server required)"
          ext=".html"
        />
        <DownloadRow href={manifestUrl} filename="manifest.json" label="Artifact Manifest" desc="JSON index of all produced artifacts with paths and sizes" ext=".json" />
      </div>

      <div className="flex gap-3">
        <a href={htmlReportUrl} target="_blank" rel="noreferrer" className="btn-primary flex-1 justify-center">
          <ExternalLink className="w-4 h-4" /> Open HTML Report
        </a>
      </div>
    </div>
  );
}

// ─── Main ResultsDashboard ────────────────────────────────────────────────

const TABS = [
  { id: 'overview', label: 'Overview', icon: BookOpen },
  { id: 'super_resolution', label: 'Super-Resolution', icon: Layers },
  { id: 'uncertainty', label: 'Uncertainty', icon: Activity },
  { id: 'segmentation', label: 'Classification', icon: Map },
  { id: 'evaluation', label: 'Metrics', icon: BarChart3 },
  { id: 'scientific', label: 'Scientific Status', icon: ShieldAlert },
  { id: 'downloads', label: 'Downloads', icon: Download },
];

export default function ResultsDashboard({ job, results, report, onReset }) {
  const [activeTab, setActiveTab] = useState('overview');
  const outputs = results.outputs || {};
  const file = path => resultFileUrl(job.job_id, path);

  const renderTab = () => {
    switch (activeTab) {
      case 'overview': return <OverviewTab job={job} report={report} outputs={outputs} file={file} />;
      case 'super_resolution': return <SuperResolutionTab file={file} outputs={outputs} />;
      case 'uncertainty': return <UncertaintyTab file={file} report={report} outputs={outputs} />;
      case 'segmentation': return <SegmentationTab file={file} report={report} outputs={outputs} />;
      case 'evaluation': return <EvaluationTab report={report} />;
      case 'scientific': return <ScientificStatusTab report={report} />;
      case 'downloads': return <DownloadsTab file={file} report={report} outputs={outputs} job={job} />;
      default: return null;
    }
  };

  return (
    <main className="min-h-screen" style={{ background: 'radial-gradient(ellipse 80% 30% at 50% 0%, rgba(6,182,212,0.08), transparent 70%), #050a14' }}>
      {/* Sticky header */}
      <header className="sticky-header px-6 md:px-12 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#0891b2,#1d4ed8)' }}>
            <CheckCircle2 className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="font-bold text-white">PixelSight</div>
            <div className="text-[10px] uppercase tracking-widest text-emerald-400">Processing complete</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 font-mono hidden sm:block">job {job.job_id}</span>
          <button className="btn-ghost" onClick={onReset}>
            <RotateCcw className="w-4 h-4" /> New image
          </button>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Tab bar */}
        <div className="flex gap-1 overflow-x-auto pb-1 mb-8 scrollbar-hide">
          {TABS.map(tab => (
            <TabButton
              key={tab.id}
              id={tab.id}
              label={tab.label}
              icon={tab.icon}
              active={activeTab === tab.id}
              onClick={setActiveTab}
            />
          ))}
        </div>

        {/* Tab content */}
        <div className="pb-16 anim-fade-up">
          {renderTab()}
        </div>
      </div>
    </main>
  );
}
