import React, { useState } from 'react';
import {
  Flame, Download, RotateCcw, AlertTriangle,
  Info, Activity, Layers, ShieldCheck, CheckCircle2,
  Calendar, ArrowRight, Clock, HelpCircle, ExternalLink,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import UncertaintyViewer from '../components/UncertaintyViewer.jsx';
import GroundTruthPage from './GroundTruthPage.jsx';
import { resultFileUrl } from '../api/srmApi.js';

export default function DisasterManagementPage({ job, results, report, onReset, onSelectApplication, health }) {
  const [activeTab, setActiveTab] = useState('slider'); // 'before' | 'after' | 'slider' | 'change' | 'reliability' | 'uncertainty' | 'ground_truth'
  const [showGtWorkspace, setShowGtWorkspace] = useState(false);

  const jobId = job?.job_id || results?.job_id;
  const disasterData = results?.outputs?.disaster || report?.disaster_analysis || {};
  const stats = disasterData?.statistics || {};
  const rel = stats?.reliability_breakdown || {};
  const alignment = results?.outputs?.alignment || report?.disaster_metadata || {};
  const gtEval = disasterData?.evaluation || results?.evaluation || {};

  // Acquisition Dates & Temporal Interval
  const beforeDate =
    alignment?.pre_date ||
    results?.outputs?.pre_date ||
    report?.input_metadata?.acquisition_date ||
    '2023-05-15';
  const afterDate =
    alignment?.post_date ||
    results?.outputs?.post_date ||
    report?.input_metadata?.post_acquisition_date ||
    '2023-06-02';

  const computeDayDiff = (d1, d2) => {
    try {
      const t1 = new Date(d1).getTime();
      const t2 = new Date(d2).getTime();
      if (!isNaN(t1) && !isNaN(t2)) {
        const diff = Math.round(Math.abs(t2 - t1) / (1000 * 60 * 60 * 24));
        return `${diff} days`;
      }
    } catch {}
    return '18 days';
  };
  const temporalDiff = computeDayDiff(beforeDate, afterDate);

  // Preview file paths
  const preSrPreview = resultFileUrl(jobId, 'application/disaster/previews/pre_sr_preview.png');
  const postSrPreview = resultFileUrl(jobId, 'application/disaster/previews/post_sr_preview.png');
  const changePreview = resultFileUrl(jobId, 'application/disaster/previews/change_preview.png');
  const affectedPreview = resultFileUrl(jobId, 'application/disaster/previews/affected_area_preview.png');
  const uncertaintyPreview = resultFileUrl(jobId, 'application/disaster/uncertainty.png');

  const downloads = [
    { name: 'Change Magnitude GeoTIFF', path: 'application/disaster/change_map.tif', desc: 'Normalized multi-spectral change vector array' },
    { name: 'Potential Affected Area Mask', path: 'application/disaster/affected_area.tif', desc: 'Binary detected change mask' },
    { name: 'Combined Uncertainty (GeoTIFF)', path: 'application/disaster/uncertainty.tif', desc: 'Maximum of pre/post diffusion variance' },
    { name: 'Pre-Event SR GeoTIFF (4x)', path: 'application/disaster/pre_event/sr_pre.tif', desc: '4x super-resolved pre-event baseline (~2.5m representation)' },
    { name: 'Post-Event SR GeoTIFF (4x)', path: 'application/disaster/post_event/sr_post.tif', desc: '4x super-resolved post-event condition (~2.5m representation)' },
    { name: 'HTML Disaster Report', path: 'report/report.html', desc: 'Standalone interactive HTML disaster assessment report' },
    { name: 'Markdown Report', path: 'report/report.md', desc: 'Scientific report with mathematical provenance' },
    { name: 'JSON Manifest', path: 'manifest.json', desc: 'Digital asset inventory and provenance manifest' },
  ];

  if (showGtWorkspace) {
    return (
      <GroundTruthPage
        initialJobId={jobId}
        onBack={() => setShowGtWorkspace(false)}
      />
    );
  }

  const isGtValidated = gtEval?.available === true && gtEval?.label_status === 'GROUND_TRUTH';

  return (
    <div className="min-h-screen pb-16 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Top Temporal Header */}
      <header className="py-5 border-b border-white/[0.08] mb-8 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-rose-500/10 border border-rose-500/30 text-rose-400">
              <Flame className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold text-white">Disaster Analysis &amp; Temporal Assessment</h1>
                <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-300">
                  Dual-Temporal Pipeline
                </span>
              </div>
              <div className="text-xs text-slate-500 mt-0.5 font-mono">
                Job ID: {jobId} · LDSR-S2 4× Diffusion Resolution (~2.5m representation)
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setShowGtWorkspace(true)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 transition-colors"
            >
              <CheckCircle2 className="w-3.5 h-3.5" /> Ground Truth Workspace
            </button>

            <button
              onClick={onReset}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 hover:text-white border border-white/[0.08] transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> New Analysis
            </button>
          </div>
        </div>

        {/* Temporal Baseline Indicator Bar */}
        <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 uppercase tracking-wider font-semibold text-[10px]">Event Assessment:</span>
            <span className="font-semibold text-slate-200">Temporal Multi-Spectral Change Detection</span>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-slate-400">
            <div className="flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" />
              <span>Before: <strong className="text-slate-200 font-mono">{beforeDate}</strong></span>
            </div>
            <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
            <div className="flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-amber-400" />
              <span>After: <strong className="text-slate-200 font-mono">{afterDate}</strong></span>
            </div>
            <span className="text-slate-600">·</span>
            <div className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-rose-400" />
              <span>Baseline: <strong className="text-slate-200 font-mono">{temporalDiff}</strong></span>
            </div>
          </div>
        </div>
      </header>

      {/* Disaster Visualization Sub-Tabs (Section 6 & 27) */}
      <div className="flex gap-2 overflow-x-auto pb-2 mb-6 border-b border-white/[0.06] text-xs font-semibold scrollbar-hide">
        <button
          onClick={() => setActiveTab('slider')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'slider'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/10'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          Before vs After (Swipe)
        </button>
        <button
          onClick={() => setActiveTab('before')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'before'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          Before Event (SR)
        </button>
        <button
          onClick={() => setActiveTab('after')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'after'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          After Event (SR)
        </button>
        <button
          onClick={() => setActiveTab('change')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'change'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          Detected Change Map
        </button>
        <button
          onClick={() => setActiveTab('reliability')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'reliability'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          Reliability Mask
        </button>
        <button
          onClick={() => setActiveTab('uncertainty')}
          className={`px-4 py-2 rounded-xl transition-all ${
            activeTab === 'uncertainty'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          Uncertainty Analysis
        </button>
      </div>

      {/* Main Grid: Visualizers on Left, Diagnostics on Right */}
      <div className="grid lg:grid-cols-[1.25fr_0.75fr] gap-8 items-start">
        {/* Left Column: Visualizers */}
        <div className="space-y-6">
          <div className="card p-5 space-y-4">
            {activeTab === 'slider' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-white flex items-center gap-1.5">
                    <Layers className="w-4 h-4 text-rose-400" />
                    Synchronized Before vs After Swipe
                  </span>
                  <span>100 LDSR-S2 diffusion sampling steps per scene</span>
                </div>
                <div className="h-[430px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40">
                  <CompareSlider
                    leftSrc={preSrPreview}
                    rightSrc={postSrPreview}
                    leftLabel={`◀ Before: ${beforeDate} (Pre-Event SR)`}
                    rightLabel={`After: ${afterDate} (Post-Event SR) ▶`}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>Interactive drag divider · Aligned to common spatial extent</span>
                  <span>~2.5m equivalent super-resolved representation</span>
                </div>
              </div>
            )}

            {activeTab === 'before' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-white">Pre-Event Baseline Acquisition</span>
                  <span className="font-mono text-cyan-300">{beforeDate}</span>
                </div>
                <div className="h-[430px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={preSrPreview} alt="Pre-Event SR" className="max-h-full max-w-full object-contain" />
                </div>
              </div>
            )}

            {activeTab === 'after' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-white">Post-Event Impacted Acquisition</span>
                  <span className="font-mono text-amber-300">{afterDate}</span>
                </div>
                <div className="h-[430px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={postSrPreview} alt="Post-Event SR" className="max-h-full max-w-full object-contain" />
                </div>
              </div>
            )}

            {activeTab === 'change' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-white">Detected Spectral Change Magnitude Map</span>
                  <span>Euclidean multi-spectral vector norm</span>
                </div>
                <div className="h-[430px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={changePreview} alt="Change Magnitude Map" className="max-h-full max-w-full object-contain" />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-400 px-1">
                  <span>Low change (navy/purple) → High change (bright yellow/orange)</span>
                  <span className="font-mono">Threshold: {stats?.change_threshold_applied ?? 0.15}</span>
                </div>
              </div>
            )}

            {activeTab === 'reliability' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-white">Uncertainty-Categorized Reliability Mask</span>
                  <span>Diffusion variance-weighted classification</span>
                </div>
                <div className="h-[430px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={affectedPreview} alt="Reliability Mask" className="max-h-full max-w-full object-contain" />
                </div>
                <div className="flex items-center gap-5 text-xs text-slate-400 px-1 pt-1">
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#28d2be]" /> Lower Uncertainty (&lt; 0.35)</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#f0be28]" /> Moderate (0.35 - 0.65)</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#eb4b4b]" /> High Uncertainty (&gt; 0.65)</span>
                </div>
              </div>
            )}

            {activeTab === 'uncertainty' && (
              <div className="space-y-4">
                <UncertaintyViewer
                  jobId={jobId}
                  uncertaintyData={disasterData?.uncertainty || report?.uncertainty}
                  outputs={results?.outputs}
                  basePath="application/disaster"
                />
              </div>
            )}
          </div>

          {/* Scientific Interpretation & Limitations (Section 7 & 27) */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Activity className="w-4 h-4 text-rose-400" />
              <span>Scientific Change Interpretation</span>
            </div>
            <div className="space-y-2 text-sm text-slate-300 leading-relaxed">
              {disasterData?.interpretations?.map((txt, idx) => (
                <div key={idx} className="flex items-start gap-2.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
                  <span>{txt}</span>
                </div>
              ))}
            </div>

            {/* Scientific Distinction: Spectral Change vs Confirmed Damage */}
            <div className="rounded-xl p-4 bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-300">
                <AlertTriangle className="w-4 h-4" />
                <span>Detected Change vs. Confirmed Damage Notice</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Detected changes represent multi-spectral radiometric variations between pre- and post-acquisitions.
                These can include phenological changes, harvest, soil moisture variations, or illumination differences,
                and are labeled as <strong className="text-amber-200">"Detected Change"</strong> rather than confirmed damage
                unless verified against validated ground truth.
              </p>
              <ul className="list-disc pl-5 space-y-1 text-slate-400 pt-1">
                {disasterData?.limitations?.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right Column: Statistics, Ground Truth, Downloads */}
        <div className="space-y-6">
          {/* Key Detected Change Metrics (Section 8 & 27) */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white">Detected Change Metrics</span>
              <span className="text-[10px] font-mono text-cyan-400 uppercase bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
                Temporal Delta
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Spectral Change Area</div>
                <div className="text-xl font-bold font-mono text-rose-300 mt-1">
                  {stats?.change_percentage != null ? stats.change_percentage.toFixed(2) : '—'}%
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">{stats?.changed_pixels ?? 0} pixels</div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Applied Threshold</div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  {stats?.change_threshold_applied != null ? stats.change_threshold_applied.toFixed(3) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Euclidean spectral norm</div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Mean Change Mag</div>
                <div className="text-xl font-bold font-mono text-slate-200 mt-1">
                  {stats?.mean_change_magnitude != null ? stats.mean_change_magnitude.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Across scene</div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Max Change Mag</div>
                <div className="text-xl font-bold font-mono text-slate-200 mt-1">
                  {stats?.max_change_magnitude != null ? stats.max_change_magnitude.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Peak anomaly</div>
              </div>
            </div>
          </div>

          {/* Uncertainty Reliability Breakdown */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Uncertainty-Aware Change Reliability</span>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Lower Uncertainty Change (&lt; 0.35)</span>
                  <span className="font-mono text-[#28d2be]">
                    {((rel?.lower_uncertainty_fraction_of_change || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-[#28d2be] rounded-full"
                    style={{ width: `${(rel?.lower_uncertainty_fraction_of_change || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Moderate Uncertainty Change</span>
                  <span className="font-mono text-[#f0be28]">
                    {((rel?.moderate_uncertainty_fraction_of_change || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-[#f0be28] rounded-full"
                    style={{ width: `${(rel?.moderate_uncertainty_fraction_of_change || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">High Uncertainty Change (&gt; 0.65)</span>
                  <span className="font-mono text-[#eb4b4b]">
                    {((rel?.high_uncertainty_fraction_of_change || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-[#eb4b4b] rounded-full"
                    style={{ width: `${(rel?.high_uncertainty_fraction_of_change || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Ground Truth Status & Evaluation (Section 9, 10, 11, 27) */}
          <div className="card p-5 space-y-4 border border-rose-500/20 bg-rose-950/[0.08]">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-rose-400" />
                Disaster Ground Truth Evaluation
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                isGtValidated
                  ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}>
                {isGtValidated ? '✓ GROUND_TRUTH' : '○ NO LABELS'}
              </span>
            </div>

            {isGtValidated ? (
              <div className="space-y-3 font-mono text-xs">
                <div className="text-xs text-emerald-300">
                  Evaluated against validated disaster ground-truth labels on common 2.5m grid.
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-white/[0.05]">
                    <span className="text-[10px] uppercase text-slate-500 block">IoU / Jaccard</span>
                    <span className="text-base font-bold text-emerald-400">{gtEval.metrics?.iou?.toFixed(4) ?? '—'}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-white/[0.05]">
                    <span className="text-[10px] uppercase text-slate-500 block">F1 / Dice</span>
                    <span className="text-base font-bold text-cyan-400">{gtEval.metrics?.f1?.toFixed(4) ?? '—'}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-white/[0.05]">
                    <span className="text-[10px] uppercase text-slate-500 block">Precision</span>
                    <span className="text-base font-bold text-slate-200">{gtEval.metrics?.precision?.toFixed(4) ?? '—'}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-white/[0.05]">
                    <span className="text-[10px] uppercase text-slate-500 block">Recall</span>
                    <span className="text-base font-bold text-slate-200">{gtEval.metrics?.recall?.toFixed(4) ?? '—'}</span>
                  </div>
                </div>

                {/* Confusion Matrix Elements */}
                {gtEval.metrics && (
                  <div className="p-3 rounded-lg bg-slate-900/90 border border-white/[0.05] space-y-1 text-[11px]">
                    <div className="text-[10px] text-slate-500 uppercase">Confusion Matrix:</div>
                    <div className="grid grid-cols-4 gap-1 text-center">
                      <div className="bg-slate-950 p-1 rounded">TP: {gtEval.metrics.tp}</div>
                      <div className="bg-slate-950 p-1 rounded">FP: {gtEval.metrics.fp}</div>
                      <div className="bg-slate-950 p-1 rounded">FN: {gtEval.metrics.fn}</div>
                      <div className="bg-slate-950 p-1 rounded">TN: {gtEval.metrics.tn}</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-slate-400 leading-relaxed">
                  Change detection without ground-truth validation.
                  Quantitative damage metrics (IoU, F1, Accuracy) are strictly withheld until independent polygon annotations are validated.
                </p>
                <button
                  onClick={() => setShowGtWorkspace(true)}
                  className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-slate-900 hover:bg-slate-800 border border-rose-500/30 text-rose-300 hover:text-white text-xs font-semibold transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5" /> Open Disaster Ground Truth Workspace
                </button>
              </div>
            )}
          </div>

          {/* Downloads Card (Section 27) */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Disaster Assessment Downloads</span>
            <div className="space-y-2">
              {downloads.map(item => (
                <a
                  key={item.name}
                  href={resultFileUrl(jobId, item.path)}
                  download
                  className="flex items-center justify-between p-3 rounded-xl bg-white/[0.025] hover:bg-white/[0.06] border border-white/[0.04] transition-colors"
                >
                  <div className="min-w-0 pr-3">
                    <div className="text-xs font-semibold text-white truncate">{item.name}</div>
                    <div className="text-[10px] text-slate-500 truncate">{item.desc}</div>
                  </div>
                  <Download className="w-4 h-4 text-rose-400 shrink-0" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
