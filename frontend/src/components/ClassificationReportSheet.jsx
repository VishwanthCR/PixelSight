import React, { useState, useMemo } from 'react';
import {
  Layers, Map, BarChart3, Grid, Image as ImageIcon,
  Download, ChevronDown, ChevronUp, Search, ArrowUpDown,
  AlertTriangle, Info, CheckCircle2, ShieldAlert, ExternalLink,
  FileSpreadsheet, FileJson
} from 'lucide-react';
import CompareSlider from './CompareSlider.jsx';

// Canonical fallback classes
const CANONICAL_CLASSES = [
  { id: 0, name: 'Tree', color: '#28b45a', desc: 'Trees & closed forest canopy' },
  { id: 1, name: 'Shrubland', color: '#78aa50', desc: 'Shrub and bush formations' },
  { id: 2, name: 'Grassland', color: '#aad264', desc: 'Natural herbaceous vegetation' },
  { id: 3, name: 'Cropland', color: '#dcbe46', desc: 'Cultivated agricultural fields' },
  { id: 4, name: 'Built-up', color: '#d25a37', desc: 'Impervious structures & building clusters' },
  { id: 5, name: 'Bare', color: '#96876e', desc: 'Bare soil, sand, and rock surfaces' },
  { id: 6, name: 'Water', color: '#327dd2', desc: 'Permanent and seasonal open water' },
];

export default function ClassificationReportSheet({
  jobId,
  classification,
  fileUrl,
  loading = false,
  error = null,
}) {
  // UI view states
  const [distMetric, setDistMetric] = useState('percent'); // 'percent' | 'ha' | 'pixels'
  const [matrixTarget, setMatrixTarget] = useState('sr'); // 'sr' | 'native'
  const [matrixViewMode, setMatrixViewMode] = useState('counts'); // 'counts' | 'percent'
  const [selectedMatrixCell, setSelectedMatrixCell] = useState(null); // { rIdx, cIdx, rName, cName, val, rowTotal, pct }
  const [selectedClassDetail, setSelectedClassDetail] = useState(null); // class object for Section 10 modal/panel
  const [mapTab, setMapTab] = useState('swipe'); // 'swipe' | 'native' | 'sr' | 'diff'
  const [searchQuery, setSearchQuery] = useState('');
  const [sortKey, setSortKey] = useState('id'); // 'id' | 'name' | 'delta_iou' | 'native_iou' | 'sr_iou' | 'support'
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc' | 'desc'
  const [expandedRow, setExpandedRow] = useState(null);

  // Collapsible section toggles
  const [showMatrix, setShowMatrix] = useState(true);
  const [showAreaComparison, setShowAreaComparison] = useState(true);
  const [showLimitations, setShowLimitations] = useState(true);
  const [showRefMeta, setShowRefMeta] = useState(true);

  if (loading) {
    return (
      <div className="card p-12 text-center space-y-3">
        <div className="w-8 h-8 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin mx-auto" />
        <div className="text-sm font-semibold text-slate-300">Loading Classification Analysis...</div>
        <p className="text-xs text-slate-500">Computing downstream land-cover comparison across native and SR grids.</p>
      </div>
    );
  }

  if (error || !classification) {
    return (
      <div className="card p-8 border border-red-500/20 bg-red-950/10 space-y-4">
        <div className="flex items-center gap-2 text-red-400 font-bold">
          <AlertTriangle className="w-5 h-5" />
          <span>Classification Data Unavailable</span>
        </div>
        <p className="text-xs text-slate-300">
          {error || 'Unable to retrieve classification report for this job. Ensure the job has finished processing.'}
        </p>
      </div>
    );
  }

  const isAvailable = classification.available !== false;
  const overall = classification.overall || {};
  const nativeO = overall.native || {};
  const srO = overall.sr || {};
  const deltaO = overall.delta || {};
  const evalMeta = classification.evaluation || {};
  const perClass = classification.per_class || [];
  const nativeDist = classification.distribution?.native || {};
  const srDist = classification.distribution?.sr || {};
  const gtDist = classification.distribution?.ground_truth || {};
  const artifacts = classification.artifacts || {};
  const predComp = classification.prediction_comparison || {};
  const confMatrixObj = classification.confusion_matrix || {};
  const interpretations = classification.interpretations || [];
  const limitations = classification.limitations || [];

  // URLs for maps/previews
  const resolveFileUrl = (p) => {
    if (!p) return '';
    if (typeof p === 'string' && (p.startsWith('http://') || p.startsWith('https://') || p.startsWith('/api/'))) return p;
    return fileUrl ? fileUrl(p) : p;
  };
  const nativeMapUrl = resolveFileUrl(artifacts.native_preview || 'application/urban/previews/segmentation_native.png');
  const srMapUrl = resolveFileUrl(artifacts.sr_preview || 'application/urban/previews/segmentation_sr.png');
  const diffMapUrl = resolveFileUrl(artifacts.difference_map || 'application/urban/previews/urban_difference.png');
  const gtMapUrl = resolveFileUrl(artifacts.ground_truth_preview || 'ground_truth/ground_truth.png');
  const gtVsNativeUrl = resolveFileUrl(artifacts.gt_vs_native || 'ground_truth/gt_vs_native.png');
  const gtVsSrUrl = resolveFileUrl(artifacts.gt_vs_sr || 'ground_truth/gt_vs_sr.png');
  const nativeVsSrUrl = resolveFileUrl(artifacts.native_vs_sr || 'ground_truth/native_vs_sr.png');

  // Label Status determination
  const labelStatus = classification.label_status || (evalMeta.has_reference_labels ? 'GROUND_TRUTH' : (classification.worldcover_proxy ? 'PROXY_LABELS' : 'NONE'));

  const renderStatusBadge = () => {
    switch (labelStatus) {
      case 'GROUND_TRUTH':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 border border-emerald-500/40 text-emerald-300">
            <span>✓</span> Ground Truth Available
          </span>
        );
      case 'REFERENCE_LABELS':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/15 border border-blue-500/40 text-blue-300">
            <span>◐</span> Reference Labels
          </span>
        );
      case 'PROXY_LABELS':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/15 border border-amber-500/40 text-amber-300">
            <span>△</span> Proxy Labels (WorldCover)
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-800 border border-slate-700 text-slate-400">
            <span>○</span> No Labels Available
          </span>
        );
    }
  };

  // Format helpers
  const fmt = (val, isPct = false, dec = 3) => {
    if (val === null || val === undefined || isNaN(val)) return '—';
    const num = Number(val);
    if (!Number.isFinite(num)) return '—';
    return isPct ? `${(num * 100).toFixed(1)}%` : num.toFixed(dec);
  };

  const fmtDelta = (val, isPct = false, dec = 3) => {
    if (val === null || val === undefined || isNaN(val)) return '—';
    const num = Number(val);
    if (!Number.isFinite(num)) return '—';
    const sign = num > 0 ? '+' : '';
    return isPct ? `${sign}${(num * 100).toFixed(1)}%` : `${sign}${num.toFixed(dec)}`;
  };

  // Sorting and filtering per-class table
  const filteredAndSortedClasses = useMemo(() => {
    let list = [...perClass];
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(c => c.name.toLowerCase().includes(q) || String(c.id).includes(q));
    }
    list.sort((a, b) => {
      let va = 0;
      let vb = 0;
      if (sortKey === 'id') { va = a.id; vb = b.id; }
      else if (sortKey === 'name') { return sortOrder === 'asc' ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name); }
      else if (sortKey === 'native_iou') { va = a.native?.iou ?? -999; vb = b.native?.iou ?? -999; }
      else if (sortKey === 'sr_iou') { va = a.sr?.iou ?? -999; vb = b.sr?.iou ?? -999; }
      else if (sortKey === 'delta_iou') { va = a.delta?.iou ?? -999; vb = b.delta?.iou ?? -999; }
      else if (sortKey === 'support') { va = a.native?.support ?? -1; vb = b.native?.support ?? -1; }
      return sortOrder === 'asc' ? va - vb : vb - va;
    });
    return list;
  }, [perClass, searchQuery, sortKey, sortOrder]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('desc');
    }
  };

  // Current active confusion matrix
  const activeConfMatrix = matrixTarget === 'native' ? confMatrixObj.native : confMatrixObj.sr;
  const hasConfusionMatrix = Array.isArray(activeConfMatrix) && activeConfMatrix.length > 0;

  return (
    <div className="space-y-8 text-slate-200">
      {/* ──────────────────────────────────────────────────────────── */}
      {/* 1. Header & Scientific Metadata (Section 3)                  */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="border-b border-white/[0.08] pb-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-3">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">CLASSIFICATION ANALYSIS</h1>
                <p className="text-xs text-slate-400 font-medium">
                  Native Input vs PixelSight LDSR-S2 Super-Resolved Representation
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {renderStatusBadge()}
          </div>
        </div>

        {labelStatus === 'PROXY_LABELS' && (
          <div className="mb-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 flex items-center gap-2">
            <Info className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>WorldCover labels are used as a proxy reference and should not be interpreted as independent ground truth.</span>
          </div>
        )}

        {/* Compact Metadata Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 pt-2 text-xs">
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Input</span>
            <span className="font-mono text-slate-200 font-semibold">10 m (Sentinel-2)</span>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">SR Output</span>
            <span className="font-mono text-cyan-300 font-semibold">4× / ~2.5 m equivalent</span>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Segmentation Model</span>
            <span className="font-mono text-slate-200 font-semibold">ESA WorldCover UNet</span>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Classes</span>
            <span className="font-mono text-slate-200 font-semibold">{classification.classes_detected || 7} classes</span>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Evaluation Grid</span>
            <span className="font-mono text-slate-200 font-semibold truncate block" title={evalMeta.grid || 'Aligned 10m grid'}>
              {evalMeta.grid || 'Aligned 10m grid'}
            </span>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/[0.05]">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Reference Source</span>
            <span className="font-mono text-slate-200 font-semibold truncate block" title={evalMeta.reference || 'None'}>
              {evalMeta.reference || 'None'}
            </span>
          </div>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 2. Summary Comparison (Section 5)                            */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              SEGMENTATION SUMMARY
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Direct metric comparison answering: "How did segmentation change after applying SR?"
            </p>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            {evalMeta.has_reference_labels ? 'Matched Evaluation Grid' : 'Prediction Comparison (No Reference)'}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-white/[0.08] text-slate-400 font-mono">
                <th className="py-2.5 pr-4 font-semibold">METRIC</th>
                <th className="py-2.5 px-4 font-semibold text-right">NATIVE INPUT (10 m)</th>
                <th className="py-2.5 px-4 font-semibold text-right text-cyan-300">PIXELSIGHT SR (~2.5 m)</th>
                <th className="py-2.5 pl-4 font-semibold text-right text-slate-300">DIFFERENCE (Δ)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono">
              <tr>
                <td className="py-2.5 pr-4 text-slate-300 font-sans font-medium">Pixel Accuracy</td>
                <td className="py-2.5 px-4 text-right">{fmt(nativeO.accuracy, true)}</td>
                <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">{fmt(srO.accuracy, true)}</td>
                <td className="py-2.5 pl-4 text-right text-slate-400">{fmtDelta(deltaO.accuracy, true)}</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-slate-300 font-sans font-medium">Mean IoU (mIoU)</td>
                <td className="py-2.5 px-4 text-right">{fmt(nativeO.miou)}</td>
                <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">{fmt(srO.miou)}</td>
                <td className="py-2.5 pl-4 text-right text-slate-400">{fmtDelta(deltaO.miou)}</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-slate-300 font-sans font-medium">Dice / Macro F1</td>
                <td className="py-2.5 px-4 text-right">{fmt(nativeO.dice)}</td>
                <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">{fmt(srO.dice)}</td>
                <td className="py-2.5 pl-4 text-right text-slate-400">{fmtDelta(deltaO.dice)}</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-slate-300 font-sans font-medium">Precision</td>
                <td className="py-2.5 px-4 text-right">{fmt(nativeO.precision)}</td>
                <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">{fmt(srO.precision)}</td>
                <td className="py-2.5 pl-4 text-right text-slate-400">{fmtDelta(deltaO.precision)}</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-slate-300 font-sans font-medium">Recall</td>
                <td className="py-2.5 px-4 text-right">{fmt(nativeO.recall)}</td>
                <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">{fmt(srO.recall)}</td>
                <td className="py-2.5 pl-4 text-right text-slate-400">{fmtDelta(deltaO.recall)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Factual non-judgmental observation */}
        <div className="p-3 rounded-lg bg-slate-900/70 border border-white/[0.04] text-xs text-slate-300 space-y-1">
          <div className="flex items-center gap-2 text-cyan-300 font-semibold text-[11px] uppercase tracking-wider">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Factual Evaluation Summary</span>
          </div>
          <p className="leading-relaxed">
            {evalMeta.has_reference_labels ? (
              deltaO.miou !== null ? (
                deltaO.miou < 0
                  ? `Native 10 m input segmentation achieved mIoU of ${fmt(nativeO.miou)}, whereas LDSR-S2 SR segmentation achieved ${fmt(srO.miou)} (Δ = ${fmtDelta(deltaO.miou)}). Metrics are evaluated on a common aligned grid.`
                  : `Native 10 m input segmentation achieved mIoU of ${fmt(nativeO.miou)}, whereas LDSR-S2 SR segmentation achieved ${fmt(srO.miou)} (Δ = ${fmtDelta(deltaO.miou)}). Metrics are evaluated on a common aligned grid.`
              ) : 'Ground-truth metrics computed across compatible evaluation grid.'
            ) : (
              'No independent ground-truth labels are available for this scene. Metrics above represent cross-resolution model consistency rather than empirical ground-truth error.'
            )}
          </p>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 3. Per-Class Report Table (Section 7)                        */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="card p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              Per-Class Classification Report
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Comprehensive IoU, F1, precision, recall, and valid support breakdown
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Search filter */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Filter class..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 rounded-lg bg-slate-900 border border-white/[0.08] text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-white/[0.08] shadow-2xl bg-slate-950/40">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="sticky top-0 z-20 bg-slate-900/95 backdrop-blur border-b border-white/[0.08] text-slate-400 font-mono whitespace-nowrap shadow-sm">
                <th
                  onClick={() => handleSort('name')}
                  className="sticky left-0 z-30 bg-slate-900/95 backdrop-blur py-3 px-4 font-semibold cursor-pointer hover:text-white shadow-[2px_0_5px_rgba(0,0,0,0.4)]"
                >
                  <div className="flex items-center gap-1.5">
                    <span>Class</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-500" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('native_iou')}
                  className="py-3 px-3 font-semibold text-right cursor-pointer hover:text-white"
                >
                  Native IoU
                </th>
                <th
                  onClick={() => handleSort('sr_iou')}
                  className="py-3 px-3 font-semibold text-right text-cyan-300 cursor-pointer hover:text-white"
                >
                  SR IoU
                </th>
                <th
                  onClick={() => handleSort('delta_iou')}
                  className="py-3 px-3 font-semibold text-right cursor-pointer hover:text-white"
                >
                  Δ IoU
                </th>
                <th className="py-3 px-3 font-semibold text-right">Native Precision</th>
                <th className="py-3 px-3 font-semibold text-right text-cyan-300">SR Precision</th>
                <th className="py-3 px-3 font-semibold text-right">Δ Precision</th>
                <th className="py-3 px-3 font-semibold text-right">Native Recall</th>
                <th className="py-3 px-3 font-semibold text-right text-cyan-300">SR Recall</th>
                <th className="py-3 px-3 font-semibold text-right">Δ Recall</th>
                <th className="py-3 px-3 font-semibold text-right">Native F1</th>
                <th className="py-3 px-3 font-semibold text-right text-cyan-300">SR F1</th>
                <th className="py-3 px-3 font-semibold text-right">Δ F1</th>
                <th
                  onClick={() => handleSort('support')}
                  className="py-3 px-4 font-semibold text-right cursor-pointer hover:text-white"
                >
                  Support
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono whitespace-nowrap">
              {filteredAndSortedClasses.map(c => {
                const nat = c.native || {};
                const sr = c.sr || {};
                const delta = c.delta || {};
                return (
                  <tr
                    key={c.id}
                    onClick={() => setSelectedClassDetail(c)}
                    className="hover:bg-cyan-950/20 cursor-pointer transition-colors group"
                    title="Click to view detailed class inspection panel"
                  >
                    <td className="sticky left-0 z-10 bg-slate-950/95 backdrop-blur py-2.5 px-4 font-sans font-medium text-slate-200 shadow-[2px_0_5px_rgba(0,0,0,0.4)] group-hover:bg-slate-900/90 transition-colors">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="w-3 h-3 rounded-md flex-shrink-0 shadow-sm"
                          style={{ backgroundColor: c.color }}
                        />
                        <span className="font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors">{c.name}</span>
                        <span className="text-[10px] text-slate-500 font-mono">#{c.id}</span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-right">{fmt(nat.iou)}</td>
                    <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{fmt(sr.iou)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{fmtDelta(delta.iou)}</td>
                    <td className="py-2.5 px-3 text-right">{fmt(nat.precision)}</td>
                    <td className="py-2.5 px-3 text-right text-cyan-300">{fmt(sr.precision)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{fmtDelta(delta.precision)}</td>
                    <td className="py-2.5 px-3 text-right">{fmt(nat.recall)}</td>
                    <td className="py-2.5 px-3 text-right text-cyan-300">{fmt(sr.recall)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{fmtDelta(delta.recall)}</td>
                    <td className="py-2.5 px-3 text-right">{fmt(nat.f1)}</td>
                    <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{fmt(sr.f1)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{fmtDelta(delta.f1)}</td>
                    <td className="py-2.5 px-4 text-right text-slate-400">{nat.support ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
          <span>Click any class row to open the in-depth class comparison panel.</span>
          <span>Horizontal scrolling enabled for wide tables.</span>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 4. Class Area / Pixel Distribution (Section 9)              */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="card p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] pb-3">
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              Class Area & Distribution
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Grouped distribution comparing Native input versus super-resolved model outputs
            </p>
          </div>

          <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setDistMetric('percent')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                distMetric === 'percent'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Area %
            </button>
            <button
              onClick={() => setDistMetric('ha')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                distMetric === 'ha'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Hectares (ha)
            </button>
            <button
              onClick={() => setDistMetric('pixels')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                distMetric === 'pixels'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Pixel Count
            </button>
          </div>
        </div>

        {/* Grouped Horizontal Bar Chart */}
        <div className="space-y-4 pt-1">
          {CANONICAL_CLASSES.map(c => {
            const nd = nativeDist[c.name] || {};
            const sd = srDist[c.name] || {};

            let nVal = nd.percent || 0;
            let sVal = sd.percent || 0;
            let maxVal = 100;
            let unit = '%';

            if (distMetric === 'ha') {
              nVal = nd.area_ha || 0;
              sVal = sd.area_ha || 0;
              const allHas = CANONICAL_CLASSES.flatMap(x => [nativeDist[x.name]?.area_ha || 0, srDist[x.name]?.area_ha || 0]);
              maxVal = Math.max(1, ...allHas);
              unit = ' ha';
            } else if (distMetric === 'pixels') {
              nVal = nd.pixel_count || 0;
              sVal = sd.pixel_count || 0;
              const allPix = CANONICAL_CLASSES.flatMap(x => [nativeDist[x.name]?.pixel_count || 0, srDist[x.name]?.pixel_count || 0]);
              maxVal = Math.max(1, ...allPix);
              unit = ' px';
            }

            const nPct = Math.min(100, Math.max(2, (nVal / maxVal) * 100));
            const sPct = Math.min(100, Math.max(2, (sVal / maxVal) * 100));

            return (
              <div key={c.id} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: c.color }} />
                    <span className="font-semibold text-slate-200">{c.name}</span>
                  </div>
                  <div className="flex items-center gap-4 font-mono text-[11px]">
                    <span className="text-slate-400">Native: {nVal}{unit}</span>
                    <span className="text-cyan-300 font-bold">SR: {sVal}{unit}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {/* Native bar */}
                  <div className="h-2.5 rounded-full bg-slate-900 overflow-hidden flex items-center">
                    <div
                      className="h-full rounded-full transition-all duration-700 bg-slate-500"
                      style={{ width: `${nPct}%` }}
                    />
                  </div>

                  {/* SR bar */}
                  <div className="h-2.5 rounded-full bg-slate-900 overflow-hidden flex items-center">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${sPct}%`, backgroundColor: c.color }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-white/[0.04] text-[11px] text-slate-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-slate-500" /> Native 10 m</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-cyan-400" /> PixelSight SR (~2.5 m)</span>
          </div>
          <span>Physical area computed from geospatial transform</span>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 5. Confusion Matrix (Section 10)                             */}
      {/* ──────────────────────────────────────────────────────────── */}
      {hasConfusionMatrix ? (
        <div className="card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div>
              <div className="flex items-center gap-2">
                <Grid className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                  Confusion Matrix
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Evaluation against reference ground-truth labels across 7 land-cover classes
              </p>
            </div>

            <div className="flex items-center gap-2">
              {/* Target toggle */}
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  onClick={() => setMatrixTarget('sr')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    matrixTarget === 'sr'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  PixelSight SR
                </button>
                <button
                  onClick={() => setMatrixTarget('native')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                    matrixTarget === 'native'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Native Input
                </button>
              </div>

              {/* View mode */}
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  onClick={() => setMatrixViewMode('counts')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                    matrixViewMode === 'counts' ? 'text-white font-bold' : 'text-slate-500'
                  }`}
                >
                  Counts
                </button>
                <button
                  onClick={() => setMatrixViewMode('percent')}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                    matrixViewMode === 'percent' ? 'text-white font-bold' : 'text-slate-500'
                  }`}
                >
                  Row %
                </button>
              </div>

              <button
                onClick={() => setShowMatrix(!showMatrix)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >
                {showMatrix ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {showMatrix && (
            <div className="overflow-x-auto rounded-lg border border-white/[0.06] p-2 bg-slate-950/60">
              <table className="w-full text-center text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/[0.08] text-slate-400 font-mono">
                    <th className="py-2 px-3 text-left font-sans font-semibold">True \ Pred</th>
                    {CANONICAL_CLASSES.map(c => (
                      <th key={c.id} className="py-2 px-2 text-center">
                        <span className="inline-block px-1.5 py-0.5 rounded text-[10px]" style={{ backgroundColor: `${c.color}25`, color: c.color }}>
                          {c.name}
                        </span>
                      </th>
                    ))}
                    <th className="py-2 px-2 font-sans font-semibold text-right text-slate-300">Total</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04] font-mono text-[11px]">
                  {activeConfMatrix.map((row, rIdx) => {
                    const rClass = CANONICAL_CLASSES[rIdx] || { name: `C${rIdx}`, color: '#94a3b8' };
                    const rowTotal = row.reduce((a, b) => a + b, 0);
                    return (
                      <tr key={rIdx} className="hover:bg-slate-900/40">
                        <td className="py-2 px-3 text-left font-sans font-medium text-slate-300 flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: rClass.color }} />
                          <span>{rClass.name}</span>
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiag = rIdx === cIdx;
                          const pct = rowTotal > 0 ? (val / rowTotal) * 100 : 0;
                          const displayVal = matrixViewMode === 'percent' ? `${pct.toFixed(1)}%` : val;
                          const cClass = CANONICAL_CLASSES[cIdx] || { name: `C${cIdx}` };
                          const isSelected = selectedMatrixCell?.rIdx === rIdx && selectedMatrixCell?.cIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              onClick={() => setSelectedMatrixCell({
                                rIdx,
                                cIdx,
                                rName: rClass.name,
                                cName: cClass.name,
                                val,
                                rowTotal,
                                pct,
                              })}
                              className={`py-2 px-2 cursor-pointer transition-all ${
                                isSelected
                                  ? 'ring-2 ring-cyan-400 bg-cyan-500/20 font-bold'
                                  : isDiag
                                  ? 'font-bold text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20'
                                  : 'text-slate-400 hover:bg-slate-800/80 hover:text-white'
                              }`}
                              title={`True: ${rClass.name} → Pred: ${cClass.name} (${val})`}
                            >
                              {displayVal}
                            </td>
                          );
                        })}
                        <td className="py-2 px-2 text-right font-bold text-slate-300">{rowTotal}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Interactive Cell Inspector (Section 8) */}
          <div className="pt-1">
            {selectedMatrixCell ? (
              <div className="flex items-center justify-between p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/30 text-xs text-slate-200 shadow-md">
                <div className="flex items-center gap-2.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="font-semibold text-slate-300">Confusion Detail:</span>
                  <span className="text-cyan-300 font-mono font-medium">
                    {selectedMatrixCell.rName === selectedMatrixCell.cName
                      ? `Correctly predicted ${selectedMatrixCell.rName} pixels: ${selectedMatrixCell.val} (${selectedMatrixCell.pct.toFixed(1)}% of total)`
                      : `${selectedMatrixCell.rName} pixels predicted as ${selectedMatrixCell.cName}: ${selectedMatrixCell.val} (${selectedMatrixCell.pct.toFixed(1)}% confusion)`}
                  </span>
                </div>
                <button
                  onClick={() => setSelectedMatrixCell(null)}
                  className="px-2 py-0.5 rounded text-[11px] text-slate-400 hover:text-white bg-white/[0.05] hover:bg-white/[0.1]"
                >
                  Clear
                </button>
              </div>
            ) : (
              <p className="text-[11px] text-slate-500 italic">
                Click a cell to inspect class confusion.
              </p>
            )}
          </div>
        </div>
      ) : predComp.cross_tabulation ? (
        /* Prediction disagreement / agreement matrix when reference labels unavailable */
        <div className="card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div>
              <div className="flex items-center gap-2">
                <Grid className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                  Model Prediction Agreement Matrix
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Cross-tabulation showing pixel-wise transition between Native 10 m and SR ~2.5 m predictions
              </p>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-cyan-500/30 text-cyan-400 bg-cyan-500/10">
              {predComp.agreement_percent}% Overall Agreement
            </span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-white/[0.06] p-2 bg-slate-950/60">
            <table className="w-full text-center text-xs border-collapse font-mono text-[11px]">
              <thead>
                <tr className="border-b border-white/[0.08] text-slate-400">
                  <th className="py-2 px-3 text-left font-sans font-semibold">Native \ SR</th>
                  {CANONICAL_CLASSES.map(c => (
                    <th key={c.id} className="py-2 px-2">
                      <span className="inline-block px-1.5 py-0.5 rounded text-[10px]" style={{ backgroundColor: `${c.color}25`, color: c.color }}>
                        {c.name}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {predComp.cross_tabulation.map((row, rIdx) => {
                  const rClass = CANONICAL_CLASSES[rIdx] || { name: `C${rIdx}`, color: '#94a3b8' };
                  return (
                    <tr key={rIdx}>
                      <td className="py-2 px-3 text-left font-sans font-medium text-slate-300 flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: rClass.color }} />
                        <span>{rClass.name}</span>
                      </td>
                      {row.map((val, cIdx) => (
                        <td
                          key={cIdx}
                          className={`py-2 px-2 ${
                            rIdx === cIdx ? 'font-bold text-cyan-300 bg-cyan-500/10' : 'text-slate-400'
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 6. Segmentation Map Comparison (Section 11 & 12)             */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="card p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-3">
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              Spatial Classification Maps
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Visual comparison between Native (10 m) and Super-Resolved (~2.5 m) semantic predictions
            </p>
          </div>

          {/* Map Tabs */}
          <div className="flex flex-wrap items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setMapTab('gt')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'gt' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              Ground Truth
            </button>
            <button
              onClick={() => setMapTab('native')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'native' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              Native
            </button>
            <button
              onClick={() => setMapTab('sr')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'sr' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              SR
            </button>
            <button
              onClick={() => setMapTab('native_vs_sr')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'native_vs_sr' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              Native vs SR
            </button>
            <button
              onClick={() => setMapTab('gt_vs_native')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'gt_vs_native' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              GT vs Native
            </button>
            <button
              onClick={() => setMapTab('gt_vs_sr')}
              className={`px-3 py-1 rounded text-[11px] font-semibold transition-all ${
                mapTab === 'gt_vs_sr' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'text-slate-400 hover:text-white'
              }`}
            >
              GT vs SR
            </button>
          </div>
        </div>

        {/* Map Display Window */}
        <div className="rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 min-h-[400px]">
          {mapTab === 'gt' && (
            <div className="p-4 flex flex-col items-center">
              {artifacts.ground_truth_preview ? (
                <>
                  <img
                    src={gtMapUrl}
                    alt="Ground Truth Label Raster"
                    className="max-h-[420px] rounded-lg object-contain shadow-lg border border-emerald-500/20"
                    onError={e => { e.target.style.display = 'none'; }}
                  />
                  <span className="text-xs text-emerald-300 mt-2 font-mono">Validated Ground-Truth Label Raster (Common Grid)</span>
                </>
              ) : (
                <div className="py-20 text-center space-y-2">
                  <div className="text-sm font-semibold text-slate-300">Ground Truth Raster Not Generated</div>
                  <p className="text-xs text-slate-500 max-w-md mx-auto">
                    No validated ground-truth raster has been created for this AOI yet. Open the Ground Truth workspace to annotate HR reference imagery.
                  </p>
                </div>
              )}
            </div>
          )}

          {mapTab === 'native' && (
            <div className="p-4 flex flex-col items-center">
              <img
                src={nativeMapUrl}
                alt="Native Segmentation Map"
                className="max-h-[420px] rounded-lg object-contain shadow-lg"
                onError={e => { e.target.style.display = 'none'; }}
              />
              <span className="text-xs text-slate-400 mt-2 font-mono">Native 10 m input segmentation</span>
            </div>
          )}

          {mapTab === 'sr' && (
            <div className="p-4 flex flex-col items-center">
              <img
                src={srMapUrl}
                alt="SR Segmentation Map"
                className="max-h-[420px] rounded-lg object-contain shadow-lg"
                onError={e => { e.target.style.display = 'none'; }}
              />
              <span className="text-xs text-cyan-300 mt-2 font-mono">LDSR-S2 4× (~2.5 m equivalent) segmentation</span>
            </div>
          )}

          {mapTab === 'native_vs_sr' && (
            <div className="p-4 flex flex-col items-center space-y-3">
              <div className="w-full h-[440px]">
                <CompareSlider
                  leftSrc={nativeMapUrl}
                  rightSrc={srMapUrl}
                  leftLabel="◀ Native Input Segmentation (10 m)"
                  rightLabel="PixelSight SR Segmentation (~2.5 m) ▶"
                />
              </div>
              <div className="flex items-center gap-4 text-xs font-mono text-slate-400 pt-1">
                <span>Swipe left/right to inspect boundary refinement and small structure delineation.</span>
              </div>
            </div>
          )}

          {mapTab === 'gt_vs_native' && (
            <div className="p-4 flex flex-col items-center space-y-3">
              {artifacts.gt_vs_native ? (
                <>
                  <img
                    src={gtVsNativeUrl}
                    alt="Ground Truth vs Native Disagreement Map"
                    className="max-h-[400px] rounded-lg object-contain shadow-lg border border-purple-500/20"
                    onError={e => { e.target.style.display = 'none'; }}
                  />
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#f0c83c]" /> Native prediction error / disagreement</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#282d37]" /> Accurate agreement with Ground Truth</span>
                  </div>
                </>
              ) : (
                <div className="py-20 text-center space-y-2">
                  <div className="text-sm font-semibold text-slate-300">Ground Truth Comparison Unavailable</div>
                  <p className="text-xs text-slate-500 max-w-md mx-auto">
                    Validated ground-truth labels are required to produce the GT vs Native spatial error map.
                  </p>
                </div>
              )}
            </div>
          )}

          {mapTab === 'gt_vs_sr' && (
            <div className="p-4 flex flex-col items-center space-y-3">
              {artifacts.gt_vs_sr ? (
                <>
                  <img
                    src={gtVsSrUrl}
                    alt="Ground Truth vs SR Disagreement Map"
                    className="max-h-[400px] rounded-lg object-contain shadow-lg border border-purple-500/20"
                    onError={e => { e.target.style.display = 'none'; }}
                  />
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#f0c83c]" /> SR prediction error / disagreement</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#282d37]" /> Accurate agreement with Ground Truth</span>
                  </div>
                </>
              ) : (
                <div className="py-20 text-center space-y-2">
                  <div className="text-sm font-semibold text-slate-300">Ground Truth Comparison Unavailable</div>
                  <p className="text-xs text-slate-500 max-w-md mx-auto">
                    Validated ground-truth labels are required to produce the GT vs SR spatial error map.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Mandatory scientific difference note */}
        <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/30 text-xs text-amber-200/90 flex items-start gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Scientific Note:</strong> Prediction differences indicate disagreement between the two segmentation outputs and should not be interpreted as observed land-cover change.
          </p>
        </div>

        {/* Centralized 7-Class Legend (Section 12) */}
        <div className="pt-2">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 block mb-2">
            Active Land-Cover Palette (ESA WorldCover 7-Class Scheme)
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
            {CANONICAL_CLASSES.map(c => (
              <div
                key={c.id}
                className="flex items-center gap-2 p-2 rounded-lg bg-slate-900/60 border border-white/[0.04] text-xs"
              >
                <span className="w-3 h-3 rounded flex-shrink-0" style={{ backgroundColor: c.color }} />
                <div className="min-w-0">
                  <span className="font-semibold text-slate-200 block truncate">{c.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono">ID {c.id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 7. Interpretation & Limitations (Section 14 & 15)           */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Interpretation */}
        <div className="card p-5 space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs uppercase tracking-wider">
            <Info className="w-4 h-4" />
            <span>Classification Interpretation</span>
          </div>
          <div className="space-y-2 text-xs text-slate-300 leading-relaxed">
            {interpretations.length > 0 ? (
              interpretations.map((item, idx) => (
                <p key={idx} className="p-2.5 rounded-lg bg-slate-900/50 border border-white/[0.04]">
                  {item}
                </p>
              ))
            ) : (
              <p className="text-slate-500 italic">No automated interpretations generated.</p>
            )}
          </div>
        </div>

        {/* Limitations */}
        <div className="card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4" />
              <span>Limitations & Interpretation Notes</span>
            </div>
            <button
              onClick={() => setShowLimitations(!showLimitations)}
              className="text-slate-400 hover:text-white"
            >
              {showLimitations ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>

          {showLimitations && (
            <ul className="space-y-1.5 text-xs text-slate-400 leading-relaxed list-disc list-inside">
              {limitations.map((lim, idx) => (
                <li key={idx} className="text-slate-300">{lim}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 8. Downloads (Section 17)                                    */}
      {/* ──────────────────────────────────────────────────────────── */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-white/[0.06] pb-3">
          <Download className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
            Classification Artifacts & Downloads
          </h3>
        </div>

        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
          <a
            href={fileUrl(artifacts.native_segmentation || 'application/urban/segmentation_native.tif')}
            download="segmentation_native.tif"
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <ImageIcon className="w-4 h-4 text-slate-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">Native Segmentation</div>
              <div className="text-[10px] text-slate-500">10 m GeoTIFF raster</div>
            </div>
          </a>

          <a
            href={fileUrl(artifacts.sr_segmentation || 'application/urban/segmentation_sr.tif')}
            download="segmentation_sr.tif"
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <ImageIcon className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">SR Segmentation</div>
              <div className="text-[10px] text-slate-500">~2.5 m equivalent GeoTIFF</div>
            </div>
          </a>

          <a
            href={`/api/v1/results/${jobId}/classification`}
            download={`classification_report_${jobId}.json`}
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <FileJson className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">Classification Report</div>
              <div className="text-[10px] text-slate-500">Structured JSON payload</div>
            </div>
          </a>

          <a
            href={`/api/v1/results/${jobId}/classification/statistics.csv`}
            download={`classification_statistics_${jobId}.csv`}
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <FileSpreadsheet className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">Class Statistics</div>
              <div className="text-[10px] text-slate-500">Per-class CSV table</div>
            </div>
          </a>

          <a
            href={`/api/v1/results/${jobId}/classification/confusion_matrix.json`}
            download={`confusion_matrix_${jobId}.json`}
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <Grid className="w-4 h-4 text-purple-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">Confusion Matrix</div>
              <div className="text-[10px] text-slate-500">7x7 matrix JSON</div>
            </div>
          </a>

          <a
            href={diffMapUrl}
            download="urban_difference.png"
            className="btn-ghost text-xs justify-start p-3 h-auto"
          >
            <ImageIcon className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-200 truncate">Difference Map</div>
              <div className="text-[10px] text-slate-500">Prediction disagreement PNG</div>
            </div>
          </a>

          {/* Ground Truth Artifact Downloads */}
          <a
            href={`/api/v1/ground-truth/${jobId}/export/geojson`}
            download={`ground_truth_${jobId}.geojson`}
            className="btn-ghost text-xs justify-start p-3 h-auto border border-emerald-500/30 bg-emerald-950/20"
          >
            <FileJson className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-emerald-300 truncate">Ground Truth GeoJSON</div>
              <div className="text-[10px] text-emerald-500/80">Vector label polygons</div>
            </div>
          </a>

          <a
            href={`/api/v1/ground-truth/${jobId}/export/geotiff`}
            download={`ground_truth_${jobId}.tif`}
            className="btn-ghost text-xs justify-start p-3 h-auto border border-emerald-500/30 bg-emerald-950/20"
          >
            <ImageIcon className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-emerald-300 truncate">Ground Truth GeoTIFF</div>
              <div className="text-[10px] text-emerald-500/80">Rasterized evaluation grid</div>
            </div>
          </a>

          <a
            href={`/api/v1/ground-truth/${jobId}/export/metadata`}
            download={`ground_truth_metadata_${jobId}.json`}
            className="btn-ghost text-xs justify-start p-3 h-auto border border-slate-700"
          >
            <FileJson className="w-4 h-4 text-slate-400 flex-shrink-0" />
            <div className="text-left min-w-0">
              <div className="font-semibold text-slate-300 truncate">GT Metadata JSON</div>
              <div className="text-[10px] text-slate-500">Validation & CRS provenance</div>
            </div>
          </a>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 9. Interactive Class Detail Modal / Panel (Section 10)        */}
      {/* ──────────────────────────────────────────────────────────── */}
      {selectedClassDetail && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm anim-fade-in"
          onClick={() => setSelectedClassDetail(null)}
        >
          <div
            className="card max-w-xl w-full p-6 space-y-5 border border-cyan-500/40 shadow-2xl relative bg-slate-950/95"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
              <div className="flex items-center gap-3">
                <span
                  className="w-4 h-4 rounded-md shadow-sm flex-shrink-0"
                  style={{ backgroundColor: selectedClassDetail.color }}
                />
                <div>
                  <h3 className="text-lg font-bold text-white uppercase tracking-wider">
                    {selectedClassDetail.name}
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">
                    Class ID: #{selectedClassDetail.id} · {CANONICAL_CLASSES.find(x => x.id === selectedClassDetail.id)?.desc || selectedClassDetail.name}
                  </span>
                </div>
              </div>
              <button
                onClick={() => setSelectedClassDetail(null)}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                title="Close"
              >
                ✕
              </button>
            </div>

            {/* Metrics comparison grid */}
            <div className={`grid grid-cols-1 sm:grid-cols-2 ${evalMeta.has_reference_labels || gtDist[selectedClassDetail.name] ? 'lg:grid-cols-4' : 'lg:grid-cols-3'} gap-3 text-xs`}>
              {(evalMeta.has_reference_labels || gtDist[selectedClassDetail.name]) && (
                <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-2">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-400 block border-b border-emerald-500/20 pb-1">
                    Ground Truth Reference
                  </span>
                  <div className="space-y-1.5 font-mono">
                    <div className="flex justify-between"><span className="text-slate-500">Area:</span> <span className="text-emerald-300 font-bold">{gtDist[selectedClassDetail.name]?.area_ha ?? '—'} ha</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Share:</span> <span className="text-emerald-300 font-bold">{gtDist[selectedClassDetail.name]?.percent ?? '—'}%</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Pixels:</span> <span className="text-slate-300">{gtDist[selectedClassDetail.name]?.count ?? selectedClassDetail.native?.support ?? '—'}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Status:</span> <span className="text-emerald-400">Validated</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Grid:</span> <span className="text-slate-300">{evalMeta.grid || 'Common'}</span></div>
                  </div>
                </div>
              )}

              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-white/[0.06] space-y-2">
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block border-b border-white/[0.05] pb-1">
                  Native Input (10 m)
                </span>
                <div className="space-y-1.5 font-mono">
                  <div className="flex justify-between"><span className="text-slate-500">IoU:</span> <span className="text-slate-200">{fmt(selectedClassDetail.native?.iou)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Precision:</span> <span className="text-slate-200">{fmt(selectedClassDetail.native?.precision)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Recall:</span> <span className="text-slate-200">{fmt(selectedClassDetail.native?.recall)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">F1 / Dice:</span> <span className="text-slate-200">{fmt(selectedClassDetail.native?.f1)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Area:</span> <span className="text-slate-200">{nativeDist[selectedClassDetail.name]?.area_ha ?? '—'} ha</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Share:</span> <span className="text-slate-200">{nativeDist[selectedClassDetail.name]?.percent ?? '—'}%</span></div>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-cyan-950/20 border border-cyan-500/30 space-y-2">
                <span className="text-[10px] uppercase font-bold tracking-wider text-cyan-400 block border-b border-cyan-500/20 pb-1">
                  PixelSight SR (~2.5 m)
                </span>
                <div className="space-y-1.5 font-mono">
                  <div className="flex justify-between"><span className="text-slate-500">IoU:</span> <span className="text-cyan-300 font-bold">{fmt(selectedClassDetail.sr?.iou)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Precision:</span> <span className="text-cyan-300">{fmt(selectedClassDetail.sr?.precision)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Recall:</span> <span className="text-cyan-300">{fmt(selectedClassDetail.sr?.recall)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">F1 / Dice:</span> <span className="text-cyan-300 font-bold">{fmt(selectedClassDetail.sr?.f1)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Area:</span> <span className="text-cyan-300">{srDist[selectedClassDetail.name]?.area_ha ?? '—'} ha</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Share:</span> <span className="text-cyan-300">{srDist[selectedClassDetail.name]?.percent ?? '—'}%</span></div>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-white/[0.06] space-y-2">
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block border-b border-white/[0.05] pb-1">
                  Difference (Δ)
                </span>
                <div className="space-y-1.5 font-mono">
                  <div className="flex justify-between"><span className="text-slate-500">Δ IoU:</span> <span className="text-slate-300">{fmtDelta(selectedClassDetail.delta?.iou)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Δ Precision:</span> <span className="text-slate-300">{fmtDelta(selectedClassDetail.delta?.precision)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Δ Recall:</span> <span className="text-slate-300">{fmtDelta(selectedClassDetail.delta?.recall)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Δ F1:</span> <span className="text-slate-300">{fmtDelta(selectedClassDetail.delta?.f1)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Δ Area:</span> <span className="text-slate-300">
                    {srDist[selectedClassDetail.name]?.percent !== undefined && nativeDist[selectedClassDetail.name]?.percent !== undefined
                      ? `${(srDist[selectedClassDetail.name].percent - nativeDist[selectedClassDetail.name].percent).toFixed(2)}% pts`
                      : '—'}
                  </span></div>
                </div>
              </div>
            </div>

            {/* Where this class occurs (Section 10) */}
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/[0.05] space-y-1.5">
              <span className="text-[11px] uppercase font-bold tracking-wider text-slate-300 block">
                Where this class occurs
              </span>
              <p className="text-xs text-slate-400 leading-relaxed">
                {CANONICAL_CLASSES.find(x => x.id === selectedClassDetail.id)?.desc || selectedClassDetail.name} features are spatially distributed across the scene.
                Inspect the <strong>Spatial Classification Maps</strong> in the workspace below to examine boundary delineation and localized super-resolution effects.
              </p>
            </div>

            <div className="flex justify-end pt-1">
              <button
                onClick={() => setSelectedClassDetail(null)}
                className="btn-ghost text-xs px-4 py-2"
              >
                Close Panel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
