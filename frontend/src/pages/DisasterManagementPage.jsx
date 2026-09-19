import React, { useState } from 'react';
import {
  Flame, Download, RotateCcw, AlertTriangle,
  Info, Activity, Layers, ShieldCheck, HelpCircle,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import { resultFileUrl } from '../api/srmApi.js';

export default function DisasterManagementPage({ job, results, report, onReset }) {
  const [activeLayer, setActiveLayer] = useState('change');
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'layer'

  const jobId = job?.job_id || results?.job_id;
  const disasterData = results?.outputs?.disaster || report?.disaster_analysis || {};
  const stats = disasterData?.statistics || {};
  const rel = stats?.reliability_breakdown || {};
  const alignment = results?.outputs?.alignment || report?.disaster_metadata || {};

  const preSrPreview = resultFileUrl(jobId, 'application/disaster/previews/pre_sr_preview.png');
  const postSrPreview = resultFileUrl(jobId, 'application/disaster/previews/post_sr_preview.png');
  const changePreview = resultFileUrl(jobId, 'application/disaster/previews/change_preview.png');
  const affectedPreview = resultFileUrl(jobId, 'application/disaster/previews/affected_area_preview.png');

  const currentPreview =
    activeLayer === 'affected'
      ? affectedPreview
      : activeLayer === 'pre'
      ? preSrPreview
      : activeLayer === 'post'
      ? postSrPreview
      : changePreview;

  const downloads = [
    { name: 'Change Magnitude GeoTIFF', path: 'application/disaster/change_map.tif', desc: 'Normalized multi-spectral change array' },
    { name: 'Potential Affected Area Mask', path: 'application/disaster/affected_area.tif', desc: 'Binary change detection mask' },
    { name: 'Combined Uncertainty (GeoTIFF)', path: 'application/disaster/uncertainty.tif', desc: 'Max of pre/post diffusion variance' },
    { name: 'Pre-Event SR GeoTIFF', path: 'application/disaster/pre_event/sr_pre.tif', desc: '4x enhanced pre-event acquisition' },
    { name: 'Post-Event SR GeoTIFF', path: 'application/disaster/post_event/sr_post.tif', desc: '4x enhanced post-event acquisition' },
    { name: 'HTML Disaster Report', path: 'report/report.html', desc: 'Standalone interactive HTML report' },
    { name: 'JSON Manifest', path: 'manifest.json', desc: 'Digital asset inventory' },
  ];

  return (
    <div className="min-h-screen pb-16 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-rose-500/10 border border-rose-500/30 text-rose-400">
            <Flame className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white">Disaster Management & Temporal Change</h1>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-300">
                Research Analysis
              </span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5 font-mono">
              Job: {jobId} · Dual-Acquisition Temporal Pipeline
            </div>
          </div>
        </div>
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Analyze Another Event
        </button>
      </header>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-8 items-start">
        {/* Left Column: Visualizers */}
        <div className="space-y-6">
          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Layers className="w-4 h-4 text-rose-400" />
                <span>Temporal Change Detection</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={() => setViewMode('slider')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'slider' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Pre vs Post Slider
                </button>
                <button
                  onClick={() => setViewMode('layer')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'layer' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Layer Selector
                </button>
              </div>
            </div>

            {viewMode === 'slider' ? (
              <div className="space-y-3">
                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40">
                  <CompareSlider
                    leftSrc={preSrPreview}
                    rightSrc={postSrPreview}
                    leftLabel="Pre-Event SR (4x)"
                    rightLabel="Post-Event SR (4x)"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>Swipe to compare pre-event state against post-event condition</span>
                  <span>Both enhanced at 100 diffusion steps</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setActiveLayer('change')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'change' ? 'bg-rose-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Change Magnitude
                  </button>
                  <button
                    onClick={() => setActiveLayer('affected')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'affected' ? 'bg-rose-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Reliability Breakdown
                  </button>
                  <button
                    onClick={() => setActiveLayer('pre')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'pre' ? 'bg-rose-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Pre-Event SR
                  </button>
                  <button
                    onClick={() => setActiveLayer('post')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'post' ? 'bg-rose-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Post-Event SR
                  </button>
                </div>
                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={currentPreview} alt="Disaster Layer" className="max-h-full max-w-full object-contain" />
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-400 px-1">
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#28d2be]" /> Lower Uncertainty</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#f0be28]" /> Moderate</span>
                  <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#eb4b4b]" /> High Uncertainty</span>
                </div>
              </div>
            )}
          </div>

          {/* Scientific Interpretation Card */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Activity className="w-4 h-4 text-rose-400" />
              <span>Temporal Change Assessment</span>
            </div>
            <div className="space-y-2 text-sm text-slate-300 leading-relaxed">
              {disasterData?.interpretations?.map((txt, idx) => (
                <div key={idx} className="flex items-start gap-2.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-2 shrink-0" />
                  <span>{txt}</span>
                </div>
              ))}
            </div>

            {/* Limitations Callout */}
            <div className="rounded-xl p-4 bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-300">
                <AlertTriangle className="w-4 h-4" />
                <span>Verification & Operational Disclaimer</span>
              </div>
              <ul className="list-disc pl-5 space-y-1 text-slate-400">
                {disasterData?.limitations?.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right Column: Statistics & Downloads */}
        <div className="space-y-6">
          {/* Key Indicators */}
          <div className="card p-5 space-y-4">
            <span className="text-sm font-semibold text-white">Detected Change Metrics</span>
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

          {/* Reliability Breakdown */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Uncertainty-Aware Reliability Breakdown</span>
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

          {/* Alignment Metadata */}
          {alignment?.transformations?.length > 0 && (
            <div className="card p-4 space-y-2 text-xs">
              <span className="font-semibold text-slate-300">Geospatial Co-Registration</span>
              <div className="text-slate-400 space-y-1">
                {alignment.transformations.map((t, idx) => (
                  <div key={idx} className="font-mono text-[11px] text-slate-400">· {t}</div>
                ))}
              </div>
            </div>
          )}

          {/* Downloads Card */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Downloadable Spatial Rasters</span>
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
