import React, { useState, useEffect } from 'react';
import {
  Download, FileText, Image as ImageIcon, RotateCcw,
  ShieldAlert, CheckCircle2, Trees, Building2,
  BarChart3, Globe2, Layers, Activity, Cpu, BookOpen,
  ChevronRight, AlertTriangle, Info, ExternalLink,
  Thermometer, Radar, Map, Sprout, LayoutDashboard,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import EvaluationMetricsPanel from '../components/EvaluationMetricsPanel.jsx';
import ClassificationSummaryCard from '../components/ClassificationSummaryCard.jsx';
import ClassificationReportSheet from '../components/ClassificationReportSheet.jsx';
import GroundTruthPage from './GroundTruthPage.jsx';
import UncertaintyViewer from '../components/UncertaintyViewer.jsx';
import { resultFileUrl, fetchJobClassification } from '../api/srmApi.js';

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
      className={`relative flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-semibold transition-all whitespace-nowrap
        ${active
          ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-[0_0_20px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/30'
          : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
        }`}
    >
      {Icon && <Icon className={`w-4 h-4 ${active ? 'text-cyan-400' : 'text-slate-400'}`} />}
      <span>{label}</span>
      {active && (
        <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full" />
      )}
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

// ─── 1. OverviewTab (Executive Summary, No Clutter) ───────────────────────

function OverviewTab({
  job,
  report,
  outputs,
  file,
  classification,
  onNavigateTab,
  activeApp = 'research',
}) {
  const [sliderBaseline, setSliderBaseline] = useState('reference');
  const evaluation = report.evaluation || {};
  const hasRefAvailable = evaluation.reference_available === true || evaluation.status === 'reference_available' || evaluation.status === 'matched';
  const hasEval = hasRefAvailable &&
    [evaluation.psnr, evaluation.ssim, evaluation.sam].some(v => v != null && Number.isFinite(Number(v)));
  const runtime = report.runtime || {};
  const sr = report.super_resolution || {};
  const input = report.input || {};
  const unc = report.uncertainty || outputs.uncertainty || {};

  const [compareMode, setCompareMode] = useState(hasRefAvailable ? 'ref_vs_sr' : 'native_vs_sr');

  useEffect(() => {
    if (hasRefAvailable) {
      setCompareMode('ref_vs_sr');
    } else {
      setCompareMode('native_vs_sr');
    }
  }, [hasRefAvailable]);

  const originalRgbUrl = file(outputs.original_preview || 'input/original_preview.png');
  const hrRefRgbUrl = file(outputs.hr_reference_preview || 'reference/aligned_reference_preview.png');
  const bicubicPreviewUrl = file('preprocessing/hr_reference_preview.png');
  const srRgbUrl = file(outputs.super_resolution_preview || 'super_resolution/sr_preview.png');

  let activeLeftUrl = originalRgbUrl;
  let activeRightUrl = srRgbUrl;
  let activeLeftLabel = '◀ INPUT · 10 m (native)';
  let activeRightLabel = 'PIXELSIGHT SR · ~2.5 m (4×) ▶';
  let activeCaption = 'Left: Native 10m Sentinel-2 input · Right: LDSR-S2 2.5m diffusion';

  if (hasRefAvailable) {
    if (compareMode === 'ref_vs_sr') {
      activeLeftUrl = hrRefRgbUrl;
      activeRightUrl = srRgbUrl;
      activeLeftLabel = '◀ HR REFERENCE · 2.5 m Ground Truth';
      activeRightLabel = 'PIXELSIGHT SR · ~2.5 m (4×) ▶';
      activeCaption = 'Left: External High-Resolution Reference (2.5m) · Right: PixelSight LDSR-S2 SR (2.5m)';
    } else if (compareMode === 'ref_vs_native') {
      activeLeftUrl = hrRefRgbUrl;
      activeRightUrl = originalRgbUrl;
      activeLeftLabel = '◀ HR REFERENCE · 2.5 m Ground Truth';
      activeRightLabel = 'NATIVE INPUT · 10 m ▶';
      activeCaption = 'Left: External High-Resolution Reference (2.5m) · Right: Native Sentinel-2 Input (10m)';
    } else {
      activeLeftUrl = originalRgbUrl;
      activeRightUrl = srRgbUrl;
      activeLeftLabel = '◀ NATIVE INPUT · 10 m (native)';
      activeRightLabel = 'PIXELSIGHT SR · ~2.5 m (4×) ▶';
      activeCaption = 'Left: Native 10m Sentinel-2 input · Right: PixelSight LDSR-S2 SR (2.5m)';
    }
  } else {
    if (compareMode === 'upscale_vs_sr') {
      activeLeftUrl = bicubicPreviewUrl;
      activeRightUrl = srRgbUrl;
      activeLeftLabel = '◀ NATIVE INPUT · 4× Display Upscale';
      activeRightLabel = 'PIXELSIGHT SR · ~2.5 m (4×) ▶';
      activeCaption = 'Left: Native Input 4× Bicubic Display Upscale · Right: LDSR-S2 Diffusion SR (2.5m)';
    } else {
      activeLeftUrl = originalRgbUrl;
      activeRightUrl = srRgbUrl;
      activeLeftLabel = '◀ NATIVE INPUT · 10 m (native)';
      activeRightLabel = 'PIXELSIGHT SR · ~2.5 m (4×) ▶';
      activeCaption = 'Left: Native 10m Sentinel-2 input · Right: LDSR-S2 2.5m diffusion';
    }
  }

  const cropData = outputs.crop || report.crop_analysis;
  const urbanData = outputs.urban || report.urban_analysis;

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Results Overview"
        subtitle={`Job ${job.job_id} — PixelSight LDSR-S2 4× Super-Resolution Pipeline`}
        badge="completed"
      />

      {/* Hero Interactive SR Visualization (Section 15 & 19) */}
      <div className="card overflow-hidden p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-xs">
          <span className="text-slate-400 font-medium">Comparison Mode:</span>
          <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
            {hasRefAvailable ? (
              <>
                <button
                  onClick={() => setCompareMode('ref_vs_sr')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    compareMode === 'ref_vs_sr'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  HR Reference ↔ SR
                </button>
                <button
                  onClick={() => setCompareMode('ref_vs_native')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    compareMode === 'ref_vs_native'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  HR Reference ↔ Native
                </button>
                <button
                  onClick={() => setCompareMode('native_vs_sr')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    compareMode === 'native_vs_sr'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Native ↔ SR
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => setCompareMode('native_vs_sr')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    compareMode === 'native_vs_sr'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Native 10m ↔ SR
                </button>
                <button
                  onClick={() => setCompareMode('upscale_vs_sr')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    compareMode === 'upscale_vs_sr'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  4× Display Upscale ↔ SR
                </button>
              </>
            )}
          </div>
        </div>

        <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40">
          <CompareSlider
            leftSrc={activeLeftUrl}
            rightSrc={activeRightUrl}
            leftLabel={activeLeftLabel}
            rightLabel={activeRightLabel}
          />
        </div>
        <div className="px-2 py-0.5 text-xs text-slate-500 flex justify-between">
          <span>{activeCaption}</span>
          <span>Scale: 10 m → ~2.5 m equivalent (4×)</span>
        </div>
      </div>

      {/* Key Processing Metrics */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Metric label="Model" value="LDSR-S2" note="Latent Diffusion Super-Resolution" />
        <Metric label="Scale" value={`${sr.scale ?? 4}×`} note="128 → 512 px tiles" />
        <Metric label="Diffusion steps" value={sr.sampling_steps ?? 100} note="per tile" />
        <Metric label="Runtime" value={`${Number(runtime.seconds || 0).toFixed(1)} s`} note={runtime.device || 'device'} />
      </div>

      {/* Compact Classification Summary Card (Only if activeApp is urban or classification) */}
      {(activeApp === 'urban' || activeApp === 'classification') && classification && (
        <ClassificationSummaryCard
          classification={classification}
          onNavigate={() => onNavigateTab('classification')}
          onNavigateGroundTruth={() => onNavigateTab('ground_truth')}
        />
      )}

      {/* High-level Application Summary Card (Crop Monitoring only) */}
      {activeApp === 'crop' && cropData && (
        <div className="card p-5 space-y-4 border border-emerald-500/20 bg-emerald-950/[0.08]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 text-emerald-400 font-semibold text-sm">
              <Sprout className="w-4 h-4" />
              <span>Vegetation & Canopy Monitoring Summary</span>
            </div>
            <button
              onClick={() => onNavigateTab('vegetation')}
              className="text-xs text-emerald-300 hover:text-white flex items-center gap-1 font-semibold transition-colors"
            >
              Open Full Vegetation Workspace →
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Native Mean NDVI</span>
              <span className="text-emerald-400 font-bold text-lg">
                {cropData.statistics?.native?.mean != null ? cropData.statistics.native.mean.toFixed(4) : '—'}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">SR Mean NDVI</span>
              <span className="text-emerald-400 font-bold text-lg">
                {cropData.statistics?.super_resolution?.mean != null ? cropData.statistics.super_resolution.mean.toFixed(4) : '—'}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Consistency MAE</span>
              <span className="text-cyan-300 font-bold text-lg">
                {cropData.consistency_metrics?.mae != null ? cropData.consistency_metrics.mae.toFixed(4) : '—'}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Canopy Stress</span>
              <span className="text-amber-400 font-bold text-lg">
                {cropData.canopy_distribution?.low_or_potential_stress_fraction != null
                  ? `${(cropData.canopy_distribution.low_or_potential_stress_fraction * 100).toFixed(1)}%`
                  : '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* High-level Application Summary Card (Urban Analysis only) */}
      {activeApp === 'urban' && urbanData && (
        <div className="card p-5 space-y-4 border border-cyan-500/20 bg-cyan-950/[0.08]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 text-cyan-400 font-semibold text-sm">
              <Building2 className="w-4 h-4" />
              <span>Urban Infrastructure Summary</span>
            </div>
            <button
              onClick={() => onNavigateTab('urban')}
              className="text-xs text-cyan-300 hover:text-white flex items-center gap-1 font-semibold transition-colors"
            >
              Open Full Urban Workspace →
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono text-xs">
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Vegetation Clusters</span>
              <span className="text-emerald-400 font-bold text-lg">
                {urbanData.trees ?? urbanData.tree_clusters ?? '—'}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Built-up Regions</span>
              <span className="text-amber-400 font-bold text-lg">
                {urbanData.houses ?? urbanData.estimated_building_clusters ?? '—'}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/70 border border-white/[0.04]">
              <span className="text-[10px] uppercase text-slate-500 block">Other Objects</span>
              <span className="text-cyan-300 font-bold text-lg">
                {urbanData.other_objects ?? '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* High-level Uncertainty Overview */}
      <div className="card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-amber-400 font-semibold">
            <Activity className="w-4 h-4" />
            <span>Generative Uncertainty (Diffusion Variance)</span>
          </div>
          <button
            onClick={() => onNavigateTab('uncertainty')}
            className="text-xs text-slate-400 hover:text-white flex items-center gap-1 font-medium transition-colors"
          >
            Inspect Uncertainty Map →
          </button>
        </div>
        <div className="grid sm:grid-cols-3 gap-3">
          <Metric label="Mean σ" value={unc.mean != null ? unc.mean.toFixed(4) : '—'} note="Normalized pixel std" />
          <Metric label="Peak σ" value={unc.peak != null ? unc.peak.toFixed(4) : '—'} note="Maximum generative variance" highlight />
          <Metric label="High Risk Pixels" value={unc.high_percent != null ? `${unc.high_percent.toFixed(1)}%` : '—'} note="Regions requiring verification" highlight />
        </div>
      </div>

      {/* High-level Evaluation Summary */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
            Quantitative Quality Summary
          </div>
          <button
            onClick={() => onNavigateTab('evaluation')}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
          >
            View Full Evaluation Suite →
          </button>
        </div>
        {hasEval ? (
          <div className="space-y-4">
            <div className="grid sm:grid-cols-3 gap-3">
              <Metric label="PSNR" value={`${Number(evaluation.psnr).toFixed(2)} dB`} note="vs HR reference" />
              <Metric label="SSIM" value={Number(evaluation.ssim).toFixed(4)} note="vs HR reference" />
              <Metric label="SAM" value={`${Number(evaluation.sam).toFixed(2)}°`} note="Spectral angle error" />
            </div>
          </div>
        ) : (
          <SciNote type="info">
            {evaluation.reason || 'Independent reference-based metrics are not available for this AOI. Self-consistency diagnostics are used.'}
          </SciNote>
        )}
      </div>

      {/* Download shortcuts */}
      <div className="grid sm:grid-cols-3 gap-3">
        <a className="btn-ghost justify-center" href={file('super_resolution/sr.tif')} download>
          <ImageIcon className="w-4 h-4 text-cyan-400" /> SR GeoTIFF
        </a>
        <a className="btn-ghost justify-center" href={file('report/report.html')} target="_blank" rel="noreferrer">
          <ExternalLink className="w-4 h-4 text-emerald-400" /> HTML Report
        </a>
        <a className="btn-ghost justify-center" href={file('report/report.json')} download="pixelsight-report.json">
          <Download className="w-4 h-4 text-slate-400" /> JSON Report
        </a>
      </div>
    </div>
  );
}

// ─── 2. VegetationTab (Dedicated Crop & Canopy Workspace) ──────────────────

function VegetationTab({ file, outputs, report }) {
  const [activeLayer, setActiveLayer] = useState('sr_ndvi');
  const [viewMode, setViewMode] = useState('slider');
  const [sliderBaseline, setSliderBaseline] = useState('reference');

  const cropData = outputs.crop || report.crop_analysis || {};
  const stats = cropData.statistics || {};
  const nativeStats = stats.native || {};
  const srStats = stats.super_resolution || {};
  const consistency = cropData.consistency_metrics || {};
  const canopy = cropData.canopy_distribution || {};

  const evaluation = report.evaluation || {};
  const hasRefAvailable = evaluation.reference_available === true || evaluation.status === 'reference_available' || evaluation.status === 'matched';

  const nativeNdviUrl = file('application/crop/previews/ndvi_native.png');
  const srNdviUrl = file('application/crop/previews/ndvi_sr.png');
  const diffPreviewUrl = file('application/crop/previews/ndvi_difference.png');
  const originalRgbUrl = file('input/original_preview.png');
  const hrRefRgbUrl = file(outputs.hr_reference_preview || 'reference/aligned_reference_preview.png');
  const srRgbUrl = file('super_resolution/sr_preview.png');

  const activeLeftUrl = sliderBaseline === 'reference' ? hrRefRgbUrl : originalRgbUrl;
  const activeLeftLabel = sliderBaseline === 'reference'
    ? (hasRefAvailable ? '◀ HR REFERENCE · 2.5 m Ground Truth' : '◀ NATIVE INPUT · 4× Display Upscale')
    : '◀ INPUT · 10 m (native)';

  const currentPreview = activeLayer === 'diff' ? diffPreviewUrl : activeLayer === 'native' ? nativeNdviUrl : srNdviUrl;

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Vegetation & Canopy Monitoring"
        subtitle="Downstream biophysical analysis computing NDVI, canopy density, and radiometric consistency across Sentinel-2 bands."
        badge="vegetation workspace"
      />

      <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-8 items-start">
        {/* Left Column: Visualizers */}
        <div className="space-y-6">
          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Layers className="w-4 h-4 text-emerald-400" />
                <span>Vegetation &amp; Surface Visualizer</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={() => setViewMode('slider')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'slider' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Swipe Comparison
                </button>
                <button
                  onClick={() => setViewMode('single')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'single' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  NDVI Layer View
                </button>
              </div>
            </div>

            {viewMode === 'slider' ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between px-1 text-xs">
                  <span className="text-slate-400 font-medium">Compare LDSR-S2 SR against:</span>
                  <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
                    <button
                      onClick={() => setSliderBaseline('reference')}
                      className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                        sliderBaseline === 'reference'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {hasRefAvailable ? 'HR Reference (2.5m)' : '4× Display Upscale'}
                    </button>
                    <button
                      onClick={() => setSliderBaseline('native')}
                      className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                        sliderBaseline === 'native'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Native 10m Input
                    </button>
                  </div>
                </div>

                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40">
                  <CompareSlider
                    leftSrc={activeLeftUrl}
                    rightSrc={srRgbUrl}
                    leftLabel={activeLeftLabel}
                    rightLabel="SR OUTPUT · ~2.5 m (4×) ▶"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>
                    {sliderBaseline === 'reference'
                      ? (hasRefAvailable ? 'Left: External HR reference (2.5m) · Right: LDSR-S2 2.5m diffusion' : 'Left: Native input 4× display upscale · Right: LDSR-S2 2.5m diffusion')
                      : 'Left: Native 10m Sentinel-2 input · Right: LDSR-S2 2.5m diffusion'}
                  </span>
                  <span>Switch to NDVI Layer View for canopy maps</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveLayer('native')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'native' ? 'bg-emerald-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Native NDVI
                  </button>
                  <button
                    onClick={() => setActiveLayer('sr_ndvi')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'sr_ndvi' ? 'bg-emerald-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    SR NDVI (4x)
                  </button>
                  <button
                    onClick={() => setActiveLayer('diff')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'diff' ? 'bg-emerald-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Difference Map
                  </button>
                </div>
                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={currentPreview} alt="NDVI Layer" className="max-h-full max-w-full object-contain" />
                </div>
              </div>
            )}
          </div>

          {/* Scientific Interpretation Card */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>Scientific Agronomical Interpretation</span>
            </div>
            <div className="space-y-2 text-sm text-slate-300 leading-relaxed">
              {cropData.interpretations?.map((txt, idx) => (
                <div key={idx} className="flex items-start gap-2.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
                  <span>{txt}</span>
                </div>
              ))}
            </div>

            {/* Limitations Callout */}
            <div className="rounded-xl p-4 bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-300">
                <AlertTriangle className="w-4 h-4" />
                <span>Limitations & Verification Disclaimer</span>
              </div>
              <ul className="list-disc pl-5 space-y-1 text-slate-400">
                {cropData.limitations?.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right Column: Radiometric Consistency & Canopy Distribution */}
        <div className="space-y-6">
          {/* Consistency Metrics Card */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white">Radiometric Consistency Metrics</span>
              <span className="text-[10px] text-slate-500 font-mono">Formula: (B08-B04)/(B08+B04)</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Native Mean NDVI</div>
                <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
                  {nativeStats.mean != null ? nativeStats.mean.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">σ = {nativeStats.std?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">SR Mean NDVI</div>
                <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
                  {srStats.mean != null ? srStats.mean.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">σ = {srStats.std?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">MAE (Consistency)</div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  {consistency.mae != null ? consistency.mae.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Mean absolute diff</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">RMSE</div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  {consistency.rmse != null ? consistency.rmse.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Root mean square error</div>
              </div>
            </div>
          </div>

          {/* Canopy Breakdown Card */}
          <div className="card p-5 space-y-4">
            <span className="text-sm font-semibold text-white">Canopy Density Distribution</span>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Potential Stress / Low Veg (&lt; 0.2)</span>
                  <span className="font-mono text-amber-300">
                    {((canopy.low_or_potential_stress_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-amber-400 rounded-full"
                    style={{ width: `${(canopy.low_or_potential_stress_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Moderate Vegetation (0.2 – 0.5)</span>
                  <span className="font-mono text-emerald-400">
                    {((canopy.moderate_vegetation_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${(canopy.moderate_vegetation_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Dense Vegetation (&gt; 0.5)</span>
                  <span className="font-mono text-emerald-300">
                    {((canopy.dense_vegetation_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-400 rounded-full"
                    style={{ width: `${(canopy.dense_vegetation_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── 3. UrbanTab (Dedicated Urban & Land-Cover Analysis) ───────────────────

function UrbanTab({ file, outputs, report }) {
  const urban = report.urban_analysis || outputs.urban || {};
  const planningMap = file(urban.map || outputs.urban_planning_map || 'application/urban/previews/segmentation_sr.png');
  const inputPlanningMap = file(outputs.input_urban_planning_map || 'application/urban/previews/segmentation_native.png');
  const builtupPreview = file('application/urban/previews/builtup_preview.png');
  const diffPreview = file('application/urban/previews/urban_difference.png');

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Urban & Infrastructure Analysis"
        subtitle="Downstream structural and land-use analysis detecting built-up boundaries and green spaces."
        badge="urban workspace"
      />

      <div className="grid sm:grid-cols-3 gap-3">
        <Metric label="Vegetation regions" value={urban.trees ?? urban.tree_clusters ?? '—'} note="Connected clusters" />
        <Metric label="Built-up regions" value={urban.houses ?? urban.estimated_building_clusters ?? '—'} note="Connected building clusters" />
        <Metric label="Other classes" value={urban.other_objects ?? '—'} note="All remaining pixels" />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card p-5 space-y-3">
          <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">PixelSight SR Segmentation</div>
          <img src={planningMap} alt="SR Segmentation preview" className="w-full h-64 object-contain rounded-lg bg-black/40" onError={e => { e.target.style.display = 'none'; }} />
        </div>
        <div className="card p-5 space-y-3">
          <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Native Input Segmentation</div>
          <img src={inputPlanningMap} alt="Native segmentation preview" className="w-full h-64 object-contain rounded-lg bg-black/40" onError={e => { e.target.style.display = 'none'; }} />
        </div>
      </div>
    </div>
  );
}

// ─── 4. UncertaintyTab ─────────────────────────────────────────────────────

function UncertaintyTab({ file, report, outputs, job }) {
  const jobId = job?.job_id || report?.job_id || outputs?.job_id;
  const uncertainty = report.uncertainty || outputs.uncertainty || {};

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Uncertainty Analysis"
        subtitle="Stochastic diffusion variation across independent LDSR-S2 sampling trajectories."
        badge="Stochastic Sampling"
      />

      <UncertaintyViewer
        jobId={jobId}
        uncertaintyData={uncertainty}
        outputs={outputs}
        basePath="uncertainty"
      />
    </div>
  );
}

// ─── 5. EvaluationTab ──────────────────────────────────────────────────────

function EvaluationTab({ report, outputs }) {
  const evaluation = report.evaluation || {};
  const uncertainty = report.uncertainty || outputs?.uncertainty || {};
  const interpretation = report.evaluation_interpretation || '';

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Image Quality Metrics"
        subtitle="PSNR, SSIM, SAM and stochastic uncertainty — comprehensive SR quality diagnostics."
        badge={evaluation.status === 'reference_available' ? 'reference_available' : 'reference_unavailable'}
      />

      <EvaluationMetricsPanel evaluation={evaluation} uncertainty={uncertainty} />

      {interpretation && (
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-3">Interpretation</div>
          <p className="text-sm text-slate-400 leading-relaxed">{interpretation}</p>
        </div>
      )}
    </div>
  );
}

// ─── 6. DownloadsTab ───────────────────────────────────────────────────────

function DownloadsTab({ file, report, outputs, job }) {
  const reportUrl = file(outputs.report || 'report/report.json');
  const htmlReportUrl = file(outputs.report_html || 'report/report.html');
  const mdReportUrl = file(outputs.report_markdown || 'report/report.md');
  const manifestUrl = file(outputs.manifest || 'manifest.json');
  const srTif = file(report.outputs?.super_resolution || 'super_resolution/sr.tif');
  const classifiedRaster = file(outputs.urban_classification || 'application/urban/segmentation_sr.tif');
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
        <DownloadRow href={classifiedRaster} filename="segmentation-sr.tif" label="Classification Raster" desc="WorldCover-proxy land cover classes (GeoTIFF)" ext=".tif" />
        <DownloadRow href={original} filename="original-preview.png" label="Original Preview PNG" desc="RGB preview of the uploaded input raster" ext=".png" />
      </div>

      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-emerald-400 font-semibold mb-4">Reports & Tables</div>
        <DownloadRow href={`/api/v1/results/${job.job_id}/classification/statistics.csv`} filename={`classification_statistics_${job.job_id}.csv`} label="Per-Class Statistics CSV" desc="Full tabular metrics per land-cover class" ext=".csv" />
        <DownloadRow href={`/api/v1/results/${job.job_id}/classification/confusion_matrix.json`} filename={`confusion_matrix_${job.job_id}.json`} label="Confusion Matrix JSON" desc="7x7 class confusion matrix" ext=".json" />
        <DownloadRow href={reportUrl} filename="pixelsight-report.json" label="JSON Report" desc="Machine-readable structured evaluation report" ext=".json" />
        <DownloadRow href={mdReportUrl} filename="pixelsight-report.md" label="Markdown Report" desc="Human-readable GitHub-flavoured Markdown report" ext=".md" />
        <DownloadRow href={htmlReportUrl} filename="pixelsight-report.html" label="HTML Report" desc="Standalone interactive HTML report" ext=".html" />
        <DownloadRow href={manifestUrl} filename="manifest.json" label="Artifact Manifest" desc="JSON index of all produced artifacts with paths and sizes" ext=".json" />
      </div>
    </div>
  );
}

// ─── Main ResultsDashboard ────────────────────────────────────────────────

export default function ResultsDashboard({
  job,
  results,
  report,
  onReset,
  onSelectApplication,
  health,
  activeApp = 'research',
}) {
  const [activeTab, setActiveTab] = useState('overview');
  const outputs = results?.outputs || {};
  const file = path => resultFileUrl(job?.job_id, path);

  const [classificationData, setClassificationData] = useState(null);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [classificationError, setClassificationError] = useState(null);

  useEffect(() => {
    if (!job?.job_id) return;
    let isMounted = true;
    setClassificationLoading(true);
    fetchJobClassification(job.job_id)
      .then(data => {
        if (isMounted) {
          setClassificationData(data);
          setClassificationLoading(false);
        }
      })
      .catch(err => {
        if (isMounted) {
          setClassificationError(err.message);
          setClassificationLoading(false);
        }
      });
    return () => { isMounted = false; };
  }, [job?.job_id]);

  const isResearchMode = activeApp === 'research' || activeApp === 'core';
  const tabs = isResearchMode
    ? [
        { id: 'overview', label: 'Overview', icon: LayoutDashboard },
        { id: 'uncertainty', label: 'Uncertainty', icon: Activity },
        { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
        { id: 'downloads', label: 'Downloads', icon: Download },
      ]
    : [
        { id: 'overview', label: 'Overview', icon: LayoutDashboard },
        { id: 'classification', label: 'Classification', icon: Layers },
        { id: 'ground_truth', label: 'Ground Truth', icon: CheckCircle2 },
        { id: 'vegetation', label: 'Vegetation', icon: Sprout },
        { id: 'urban', label: 'Urban', icon: Building2 },
        { id: 'uncertainty', label: 'Uncertainty', icon: Activity },
        { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
        { id: 'downloads', label: 'Downloads', icon: Download },
      ];

  const renderTab = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewTab
            job={job}
            report={report}
            outputs={outputs}
            file={file}
            classification={classificationData}
            onNavigateTab={setActiveTab}
            activeApp={activeApp}
          />
        );
      case 'classification':
        return (
          <ClassificationReportSheet
            jobId={job.job_id}
            classification={classificationData}
            fileUrl={file}
            loading={classificationLoading}
            error={classificationError}
          />
        );
      case 'ground_truth':
        return (
          <GroundTruthPage
            initialJobId={job.job_id}
            onBack={() => setActiveTab('classification')}
          />
        );
      case 'vegetation':
        return <VegetationTab file={file} outputs={outputs} report={report} />;
      case 'urban':
        return <UrbanTab file={file} outputs={outputs} report={report} />;
      case 'uncertainty':
        return <UncertaintyTab file={file} report={report} outputs={outputs} job={job} />;
      case 'evaluation':
        return <EvaluationTab report={report} outputs={outputs} />;
      case 'downloads':
        return <DownloadsTab file={file} report={report} outputs={outputs} job={job} />;
      default:
        return null;
    }
  };

  return (
    <main className="min-h-screen" style={{ background: 'radial-gradient(ellipse 80% 30% at 50% 0%, rgba(6,182,212,0.08), transparent 70%), #050a14' }}>
      {/* Sticky Top Header */}
      <header className="sticky-header px-6 md:px-12 py-4 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#0891b2,#1d4ed8)' }}>
            <CheckCircle2 className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="font-bold text-white flex items-center gap-2">
              PixelSight Results Studio
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase font-mono">
                {activeApp || 'Pipeline'}
              </span>
            </div>
            <div className="text-[10px] uppercase tracking-widest text-emerald-400">
              Processing complete · Job {job?.job_id}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {onSelectApplication && (
            <div className="flex items-center bg-slate-900/90 rounded-xl p-1 border border-slate-800 text-xs">
              <button
                onClick={() => onSelectApplication('crop')}
                className="px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-white font-medium transition-colors"
              >
                Crop
              </button>
              <button
                onClick={() => onSelectApplication('urban')}
                className="px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-white font-medium transition-colors"
              >
                Urban
              </button>
              <button
                onClick={() => onSelectApplication('disaster')}
                className="px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-white font-medium transition-colors"
              >
                Disaster
              </button>
            </div>
          )}

          <button className="btn-ghost" onClick={onReset}>
            <RotateCcw className="w-4 h-4" /> New Image / AOI
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Top-Level Results Navigation Bar (Section 1) */}
        <nav
          aria-label="Results navigation"
          className="flex gap-2 overflow-x-auto pb-2 mb-8 border-b border-white/[0.08] scrollbar-hide"
        >
          {tabs.map(tab => (
            <TabButton
              key={tab.id}
              id={tab.id}
              label={tab.label}
              icon={tab.icon}
              active={activeTab === tab.id}
              onClick={setActiveTab}
            />
          ))}
        </nav>

        {/* Tab content area — replaced completely when switching tabs (Section 1) */}
        <div className="pb-16 anim-fade-up">
          {renderTab()}
        </div>
      </div>
    </main>
  );
}
