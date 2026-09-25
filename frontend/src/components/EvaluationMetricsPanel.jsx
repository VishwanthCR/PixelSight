/**
 * EvaluationMetricsPanel — Rich interactive visualization for PixelSight
 * quality metrics (PSNR, SSIM, SAM, uncertainty) with gauge rings, trend
 * bars, and scientific context annotations.
 */
import React, { useEffect, useRef } from 'react';
import { Info, AlertTriangle, CheckCircle2, TrendingUp, TrendingDown, Minus } from 'lucide-react';

// ── Gauge Ring (SVG arc drawn in React) ─────────────────────────────────────
function GaugeRing({ value, max, color, size = 88, strokeWidth = 8, label, unit }) {
  const r = (size - strokeWidth) / 2;
  const circ = 2 * Math.PI * r;
  const percent = Math.min(1, Math.max(0, value / max));
  const dashOffset = circ * (1 - percent);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Track */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={dashOffset}
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: 'center',
            filter: `drop-shadow(0 0 6px ${color}80)`,
            transition: 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)',
          }}
        />
        {/* Center text */}
        <text
          x={size / 2} y={size / 2 - 4}
          textAnchor="middle" dominantBaseline="middle"
          fill="white" fontSize={size < 80 ? 11 : 13} fontWeight="700"
          fontFamily="ui-monospace, monospace"
        >
          {value == null ? '—' : typeof value === 'number' ? value.toFixed(value >= 10 ? 1 : 3) : value}
        </text>
        {unit && (
          <text
            x={size / 2} y={size / 2 + 13}
            textAnchor="middle" dominantBaseline="middle"
            fill="rgba(148,163,184,0.7)" fontSize={9} fontFamily="system-ui"
          >
            {unit}
          </text>
        )}
      </svg>
      <div className="text-[11px] font-semibold text-slate-400 tracking-wide uppercase">{label}</div>
    </div>
  );
}

// ── Horizontal bar metric ────────────────────────────────────────────────────
function BarMetric({ label, value, max, color, note, suffix = '' }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-xs">
        <span className="text-slate-400 font-semibold">{label}</span>
        <span className="font-mono font-bold text-white">
          {value == null ? '—' : typeof value === 'number' ? value.toFixed(value >= 10 ? 2 : 4) : value}
          {suffix}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{
            width: `${pct}%`,
            background: color,
            boxShadow: `0 0 8px ${color}60`,
          }}
        />
      </div>
      {note && <div className="text-[10px] text-slate-600 leading-relaxed">{note}</div>}
    </div>
  );
}

// ── Trend badge ──────────────────────────────────────────────────────────────
function TrendBadge({ value, higherIsBetter = true }) {
  if (value == null || !Number.isFinite(value)) return null;
  const positive = higherIsBetter ? value > 0 : value < 0;
  const neutral = Math.abs(value) < 0.001;
  if (neutral) return <span className="text-slate-500 text-[10px] flex items-center gap-0.5"><Minus className="w-3 h-3" />—</span>;
  return (
    <span className={`text-[10px] flex items-center gap-0.5 font-mono ${positive ? 'text-emerald-400' : 'text-red-400'}`}>
      {positive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
      {Math.abs(value).toFixed(2)}
    </span>
  );
}

// ── Scientific note ──────────────────────────────────────────────────────────
function SciAnnotation({ type = 'info', children }) {
  const styles = {
    info: 'bg-sky-500/10 border-sky-500/30 text-sky-200/80',
    warning: 'bg-amber-500/10 border-amber-500/30 text-amber-200/80',
    error: 'bg-red-500/10 border-red-500/30 text-red-200/80',
  };
  const Icons = { info: Info, warning: AlertTriangle, error: AlertTriangle };
  const Icon = Icons[type] || Icons.info;
  return (
    <div className={`rounded-xl p-3.5 border text-xs leading-relaxed flex gap-2.5 ${styles[type]}`}>
      <Icon className="w-3.5 h-3.5 shrink-0 mt-0.5 opacity-70" />
      <div>{children}</div>
    </div>
  );
}

// ── Metric range reference ───────────────────────────────────────────────────
function QualityBand({ value, bands }) {
  const band = bands.find(b => value >= b.min && value <= b.max) || bands[bands.length - 1];
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${band.style}`}>
      {band.label}
    </span>
  );
}

const PSNR_BANDS = [
  { min: 0,  max: 20, label: 'Poor',      style: 'bg-red-900/40 text-red-400' },
  { min: 20, max: 28, label: 'Fair',      style: 'bg-amber-900/40 text-amber-400' },
  { min: 28, max: 35, label: 'Good',      style: 'bg-emerald-900/40 text-emerald-400' },
  { min: 35, max: 999, label: 'Excellent', style: 'bg-cyan-900/40 text-cyan-400' },
];
const SSIM_BANDS = [
  { min: 0,    max: 0.6,  label: 'Poor',      style: 'bg-red-900/40 text-red-400' },
  { min: 0.6,  max: 0.8,  label: 'Fair',      style: 'bg-amber-900/40 text-amber-400' },
  { min: 0.8,  max: 0.95, label: 'Good',      style: 'bg-emerald-900/40 text-emerald-400' },
  { min: 0.95, max: 1.0,  label: 'Excellent', style: 'bg-cyan-900/40 text-cyan-400' },
];
const SAM_BANDS = [
  { min: 0,   max: 3,   label: 'Excellent', style: 'bg-cyan-900/40 text-cyan-400' },
  { min: 3,   max: 8,   label: 'Good',      style: 'bg-emerald-900/40 text-emerald-400' },
  { min: 8,   max: 15,  label: 'Fair',      style: 'bg-amber-900/40 text-amber-400' },
  { min: 15,  max: 999, label: 'Poor',      style: 'bg-red-900/40 text-red-400' },
];

// ── Main Component ───────────────────────────────────────────────────────────
export default function EvaluationMetricsPanel({ evaluation, uncertainty }) {
  const eval_ = evaluation || {};
  const unc = uncertainty || {};
  const hasEval = eval_.status === 'reference_available' &&
    [eval_.psnr, eval_.ssim, eval_.sam].every(v => Number.isFinite(Number(v)));

  const psnr = hasEval ? Number(eval_.psnr) : null;
  const ssim = hasEval ? Number(eval_.ssim) : null;
  const sam  = hasEval ? Number(eval_.sam)  : null;

  const uncMean = Number(unc.mean_uncertainty ?? unc.mean ?? 0);
  const uncMax  = Number(unc.max_uncertainty  ?? unc.max  ?? 0);
  const uncP95  = Number(unc.p95_uncertainty  ?? unc.p95  ?? 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          Evaluation Metrics
          {hasEval ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
              No reference available
            </span>
          )}
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Self-consistency diagnostics: input (resampled) vs LDSR-S2 SR output
        </p>
      </div>

      {/* Gauge rings — main 3 metrics */}
      {hasEval ? (
        <>
          <div className="flex justify-around items-end p-5 rounded-2xl bg-slate-900/80 border border-slate-800">
            <div className="text-center">
              <GaugeRing value={psnr} max={50} color="#06b6d4" label="PSNR" unit="dB" />
              <div className="mt-1.5">
                <QualityBand value={psnr} bands={PSNR_BANDS} />
              </div>
            </div>
            <div className="text-center">
              <GaugeRing value={ssim} max={1} color="#10b981" label="SSIM" unit="0–1" />
              <div className="mt-1.5">
                <QualityBand value={ssim} bands={SSIM_BANDS} />
              </div>
            </div>
            <div className="text-center">
              <GaugeRing value={sam} max={20} color="#f59e0b" label="SAM" unit="deg" />
              <div className="mt-1.5">
                <QualityBand value={sam} bands={SAM_BANDS} />
              </div>
            </div>
          </div>

          {/* Bar breakdowns */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">Metric Breakdown</div>
            <BarMetric
              label="Peak Signal-to-Noise Ratio (PSNR)"
              value={psnr} max={50} suffix=" dB"
              color="linear-gradient(90deg, #0891b2, #38bdf8)"
              note="Higher is better. Typical bicubic upsampling: 28–33 dB on Sentinel-2 data."
            />
            <BarMetric
              label="Structural Similarity Index (SSIM)"
              value={ssim} max={1}
              color="linear-gradient(90deg, #059669, #34d399)"
              note="Range 0–1. Values ≥ 0.85 indicate good structural preservation."
            />
            <BarMetric
              label="Spectral Angle Mapper (SAM)"
              value={sam} max={20} suffix="°"
              color="linear-gradient(90deg, #d97706, #fbbf24)"
              note="Lower is better. Measures per-pixel spectral divergence in degrees."
            />
          </div>

          <SciAnnotation type="warning">
            <strong>Scientific context:</strong> These metrics compare the input (resampled to the SR grid) against the generated SR output.
            They are self-consistency diagnostics, <em>not</em> validation against independent high-resolution ground truth.
            PixelSight research establishes that LDSR does not outperform bicubic baselines on standardized metrics —
            its value lies in perceptual detail and downstream task performance.
          </SciAnnotation>
        </>
      ) : (
        <SciAnnotation type="info">
          {eval_.reason ||
            'Reference-based metrics require a high-resolution reference image for comparison. No independent ground truth was provided.'}
        </SciAnnotation>
      )}

      {/* Uncertainty metrics */}
      {(uncMean > 0 || uncMax > 0) && (
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-1">
            Stochastic Uncertainty
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { l: 'Mean σ',  v: uncMean, color: '#8b5cf6', fmt: v => v.toFixed(4) },
              { l: 'Max σ',   v: uncMax,  color: '#ec4899', fmt: v => v.toFixed(4) },
              { l: 'P95 σ',   v: uncP95,  color: '#f43f5e', fmt: v => v.toFixed(4) },
            ].map(({ l, v, color, fmt }) => (
              <div key={l} className="text-center p-3 rounded-xl bg-slate-800/60 border border-slate-700">
                <div className="font-mono text-sm font-bold" style={{ color }}>{fmt(v)}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">{l}</div>
              </div>
            ))}
          </div>
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000"
              style={{
                width: `${Math.min(100, uncMean * 1000)}%`,
                background: 'linear-gradient(90deg, #7c3aed, #ec4899)',
              }}
            />
          </div>
          <SciAnnotation type="info">
            Uncertainty is estimated via Monte Carlo dropout across multiple forward passes.
            Higher uncertainty regions indicate where the diffusion model is most ambiguous —
            treat these pixels with additional caution in downstream analysis.
          </SciAnnotation>
        </div>
      )}
    </div>
  );
}
