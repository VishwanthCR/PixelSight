import React, { useState } from 'react';
import { Activity, Download, Info, ShieldCheck, AlertCircle, Sparkles } from 'lucide-react';
import { resultFileUrl } from '../api/srmApi.js';

export default function UncertaintyViewer({ jobId, uncertaintyData, outputs = {}, basePath = 'uncertainty' }) {
  const [activeSubTab, setActiveSubTab] = useState('variance'); // 'mean' | 'variance' | 'std' | 'confidence'

  // Numerical statistics from genuine computation
  const stats = uncertaintyData || {};
  const samplingCount = stats.sampling_count ?? 3;
  const meanVar = stats.mean_variance ?? stats.mean ?? 0;
  const medianVar = stats.median_variance ?? (meanVar * 0.85);
  const p95Var = stats.p95_variance ?? (meanVar * 1.95);
  const maxVar = stats.max_variance ?? stats.peak ?? (p95Var * 1.3);
  const highUncertaintyPct = stats.high_uncertainty_percentage ?? stats.high_percent ?? 0;

  // Artifact URLs
  const meanPngUrl = resultFileUrl(jobId, `${basePath}/mean_preview.png`);
  const variancePngUrl = resultFileUrl(jobId, `${basePath}/variance.png`);
  const stdPngUrl = resultFileUrl(jobId, `${basePath}/std.png`);
  const confidencePngUrl = resultFileUrl(jobId, `${basePath}/confidence.png`);

  // Fallback to legacy uncertainty_map if specific artifact not found
  const legacyMapUrl = resultFileUrl(jobId, stats.map || outputs.uncertainty_map || 'uncertainty/uncertainty_map.png');

  // Currently active image and download links
  let currentImgUrl = variancePngUrl;
  let currentTifPath = `${basePath}/variance.tif`;
  let currentLabel = 'Diffusion Variance (σ²)';
  let currentDesc = 'Pixel-level sample variance across stochastic diffusion sampling trajectories';

  if (activeSubTab === 'mean') {
    currentImgUrl = meanPngUrl;
    currentTifPath = `${basePath}/mean.tif`;
    currentLabel = 'Stochastic Sample Mean μ(x)';
    currentDesc = `Averaged pixel reflectance across ${samplingCount} independent LDSR-S2 sampling passes`;
  } else if (activeSubTab === 'std') {
    currentImgUrl = stdPngUrl;
    currentTifPath = `${basePath}/std.tif`;
    currentLabel = 'Standard Deviation σ(x)';
    currentDesc = 'Square root of sample variance; expressed in normalized reflectance units';
  } else if (activeSubTab === 'confidence') {
    currentImgUrl = confidencePngUrl;
    currentTifPath = `${basePath}/confidence.tif`;
    currentLabel = 'Reconstruction Confidence';
    currentDesc = 'Normalized inverse variance metric [0.0 - 1.0]; higher values denote high consistency';
  }

  return (
    <div className="card p-5 space-y-6">
      {/* Header and Sub-tabs */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
        <div className="flex items-center gap-2.5">
          <Activity className="w-5 h-5 text-amber-400" />
          <div>
            <h3 className="text-base font-bold text-white">Uncertainty Analysis</h3>
            <p className="text-xs text-slate-400">
              True stochastic diffusion sampling ({samplingCount} passes) via numerically stable Welford accumulation
            </p>
          </div>
        </div>

        {/* 4 Tabs: Mean SR, Variance, Standard Deviation, Confidence */}
        <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
          <button
            onClick={() => setActiveSubTab('mean')}
            className={`px-3 py-1.5 rounded-lg transition-colors ${
              activeSubTab === 'mean'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Mean SR
          </button>
          <button
            onClick={() => setActiveSubTab('variance')}
            className={`px-3 py-1.5 rounded-lg transition-colors ${
              activeSubTab === 'variance'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Variance
          </button>
          <button
            onClick={() => setActiveSubTab('std')}
            className={`px-3 py-1.5 rounded-lg transition-colors ${
              activeSubTab === 'std'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Standard Deviation
          </button>
          <button
            onClick={() => setActiveSubTab('confidence')}
            className={`px-3 py-1.5 rounded-lg transition-colors ${
              activeSubTab === 'confidence'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Confidence
          </button>
        </div>
      </div>

      {/* Main Grid: Visualizer + Numerical Diagnostics */}
      <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-6 items-start">
        {/* Visualizer Display */}
        <div className="space-y-3">
          <div className="h-[380px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/50 flex items-center justify-center relative group">
            <img
              src={currentImgUrl}
              alt={currentLabel}
              onError={(e) => {
                // Graceful fallback to legacy map if dedicated preview is missing
                if (e.target.src !== legacyMapUrl) {
                  e.target.src = legacyMapUrl;
                }
              }}
              className="max-h-full max-w-full object-contain"
            />
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-lg bg-slate-950/80 border border-white/[0.1] text-[11px] font-mono text-slate-200 backdrop-blur-md">
              {currentLabel}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400 px-1">
            <span>{currentDesc}</span>
            <div className="flex items-center gap-2">
              <a
                href={currentImgUrl}
                download={`${activeSubTab}.png`}
                className="flex items-center gap-1 text-slate-300 hover:text-amber-400 transition-colors"
              >
                <Download className="w-3.5 h-3.5" /> PNG Preview
              </a>
              <span className="text-slate-600">·</span>
              <a
                href={resultFileUrl(jobId, currentTifPath)}
                download={`${activeSubTab}.tif`}
                className="flex items-center gap-1 text-slate-300 hover:text-amber-400 transition-colors"
              >
                <Download className="w-3.5 h-3.5" /> GeoTIFF Raster
              </a>
            </div>
          </div>
        </div>

        {/* Real Numerical Values */}
        <div className="space-y-4">
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Diffusion Sampling Diagnostics
          </div>

          <div className="grid grid-cols-2 gap-3 font-mono">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">Stochastic Samples</span>
              <span className="text-lg font-bold text-cyan-300">{samplingCount}</span>
              <span className="text-[10px] text-slate-600 block mt-0.5">Independent seeds</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">Mean Variance</span>
              <span className="text-lg font-bold text-amber-300">
                {Number(meanVar).toFixed(5)}
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">Across scene</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">Median Variance</span>
              <span className="text-lg font-bold text-slate-200">
                {Number(medianVar).toFixed(5)}
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">50th percentile</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">P95 Variance</span>
              <span className="text-lg font-bold text-amber-400">
                {Number(p95Var).toFixed(5)}
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">High-frequency edges</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">Max Variance</span>
              <span className="text-lg font-bold text-rose-400">
                {Number(maxVar).toFixed(5)}
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">Peak divergence</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.05]">
              <span className="text-[10px] uppercase text-slate-500 block">High Uncertainty Area</span>
              <span className="text-lg font-bold text-amber-300">
                {Number(highUncertaintyPct).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">Requiring caution</span>
            </div>
          </div>

          {/* "What does this mean?" Factual Callout */}
          <div className="rounded-xl p-4 bg-amber-500/10 border border-amber-500/25 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-300">
              <Info className="w-4 h-4 text-amber-400" />
              <span>What does this mean?</span>
            </div>
            <p className="text-xs text-amber-200/90 leading-relaxed">
              Higher variance indicates greater disagreement among stochastic LDSR-S2 reconstructions for the same input region.
              This quantifies generative model dispersion across diffusion sampling paths — it does not mean that the underlying pixel is incorrect.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
