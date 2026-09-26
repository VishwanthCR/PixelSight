import React from 'react';
import { ArrowRight, Layers, MapPin, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ClassificationSummaryCard({ classification, onNavigate, onNavigateGroundTruth }) {
  if (!classification) {
    return (
      <div className="card p-5 border border-white/[0.08] bg-slate-950/40">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Classification</span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-slate-700 text-slate-400">
            loading...
          </span>
        </div>
        <div className="py-6 text-center text-xs text-slate-500">
          Loading classification metrics...
        </div>
      </div>
    );
  }

  const isAvailable = classification.available !== false;
  const overall = classification.overall || {};
  const native = overall.native || {};
  const sr = overall.sr || {};
  const delta = overall.delta || {};
  const evalMeta = classification.evaluation || {};
  const classesDetected = classification.classes_detected ?? '—';
  const labelStatus = classification.label_status || evalMeta.label_status || (evalMeta.has_reference_labels ? 'PROXY_LABELS' : 'NONE');

  // Format helpers
  const fmt = (val, isPct = false) => {
    if (val === null || val === undefined || isNaN(val)) return '—';
    const num = Number(val);
    if (!Number.isFinite(num)) return '—';
    return isPct ? `${(num * 100).toFixed(1)}%` : num.toFixed(3);
  };

  // Label status badge
  let badgeLabel = '○ No Labels';
  let badgeStyle = 'border-slate-700 text-slate-400 bg-slate-900/60';
  if (labelStatus === 'GROUND_TRUTH') {
    badgeLabel = '✓ Ground Truth Available';
    badgeStyle = 'border-emerald-500/40 text-emerald-300 bg-emerald-500/10 shadow-[0_0_12px_rgba(16,185,129,0.15)]';
  } else if (labelStatus === 'REFERENCE_LABELS') {
    badgeLabel = '◐ Reference Labels';
    badgeStyle = 'border-amber-500/40 text-amber-300 bg-amber-500/10';
  } else if (labelStatus === 'PROXY_LABELS') {
    badgeLabel = '△ Proxy Labels';
    badgeStyle = 'border-sky-500/40 text-sky-300 bg-sky-500/10';
  }

  // Factual, scientifically grounded status text
  let statusLine = 'Segmentation results available for native and SR imagery.';
  if (!isAvailable) {
    statusLine = classification.reason || 'Classification analysis not executed for this job.';
  } else if (labelStatus === 'GROUND_TRUTH') {
    if (delta.miou !== null && delta.miou !== undefined) {
      if (delta.miou < 0) {
        statusLine = 'In this evaluation, SR segmentation mIoU was lower than the native segmentation.';
      } else if (delta.miou > 0) {
        statusLine = 'In this evaluation, SR segmentation mIoU was higher than the native segmentation.';
      } else {
        statusLine = 'Native and SR segmentation mIoU are equivalent in this evaluation.';
      }
    } else {
      statusLine = 'Evaluated against validated expert ground-truth annotations on common evaluation grid.';
    }
  } else if (labelStatus === 'PROXY_LABELS') {
    statusLine = 'WorldCover labels are used as a proxy reference and should not be interpreted as independent ground truth.';
  } else {
    statusLine = 'Model prediction comparison available; no ground-truth reference labels for this AOI.';
  }

  return (
    <div className="card p-5 border border-white/[0.08] bg-slate-950/40 space-y-4 hover:border-cyan-500/20 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs uppercase tracking-wider text-slate-300 font-bold">Classification</h3>
        </div>
        <span className={`text-[10px] font-mono font-semibold px-2.5 py-0.5 rounded-full border ${badgeStyle}`}>
          {badgeLabel}
        </span>
      </div>

      {/* Metrics 2x2 Grid */}
      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="p-3 rounded-lg bg-slate-900/60 border border-white/[0.04]">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">Input mIoU</div>
          <div className="text-xl font-bold font-mono text-slate-200 mt-1">
            {fmt(native.miou)}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {labelStatus === 'GROUND_TRUTH' ? 'vs Ground Truth' : 'Native 10 m'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900/60 border border-white/[0.04]">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">SR mIoU</div>
          <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
            {fmt(sr.miou)}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {labelStatus === 'GROUND_TRUTH' ? 'vs Ground Truth' : '~2.5 m equivalent'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900/60 border border-white/[0.04]">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">Input Accuracy</div>
          <div className="text-xl font-bold font-mono text-slate-200 mt-1">
            {fmt(native.accuracy, true)}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {evalMeta.grid?.includes('2.5') ? '2.5 m grid' : 'Matched grid'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900/60 border border-white/[0.04]">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">SR Accuracy</div>
          <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
            {fmt(sr.accuracy, true)}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {evalMeta.grid?.includes('2.5') ? '2.5 m grid' : 'Matched grid'}
          </div>
        </div>
      </div>

      {/* Footer / Status Info */}
      <div className="pt-1 border-t border-white/[0.05] space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>Classes detected:</span>
          <span className="font-mono text-slate-200 font-semibold">{classesDetected}</span>
        </div>

        <p className="text-[11px] text-slate-400 leading-relaxed italic">
          "{statusLine}"
        </p>

        <div className="flex flex-wrap items-center gap-2 mt-2">
          <button
            onClick={onNavigate}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 transition-all group"
          >
            <span>View Classification Report</span>
            <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
          </button>
          {onNavigateGroundTruth && (
            <button
              onClick={onNavigateGroundTruth}
              className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 transition-all"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Ground Truth</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
