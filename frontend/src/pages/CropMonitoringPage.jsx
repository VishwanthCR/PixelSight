import React, { useState } from 'react';
import {
  Sprout, Download, FileText, RotateCcw, AlertTriangle,
  Info, Activity, ShieldAlert, Layers, CheckCircle2,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import { resultFileUrl } from '../api/srmApi.js';

export default function CropMonitoringPage({ job, results, report, onReset }) {
  const [activeLayer, setActiveLayer] = useState('sr_ndvi');
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'single'

  const jobId = job?.job_id || results?.job_id;
  const cropData = results?.outputs?.crop || report?.crop_analysis || {};
  const stats = cropData?.statistics || {};
  const nativeStats = stats?.native || {};
  const srStats = stats?.super_resolution || {};
  const consistency = cropData?.consistency_metrics || {};
  const canopy = cropData?.canopy_distribution || {};
  const unc = results?.outputs?.uncertainty || report?.uncertainty || {};

  const nativeNdviUrl  = resultFileUrl(jobId, 'application/crop/previews/ndvi_native.png');
  const srNdviUrl      = resultFileUrl(jobId, 'application/crop/previews/ndvi_sr.png');
  const diffPreviewUrl = resultFileUrl(jobId, 'application/crop/previews/ndvi_difference.png');
  const originalRgbUrl = resultFileUrl(jobId, 'input/original_preview.png');
  const srRgbUrl       = resultFileUrl(jobId, 'super_resolution/sr_preview.png');

  // Layer View shows NDVI products; Swipe View shows the raw image quality difference
  const currentPreview = activeLayer === 'diff' ? diffPreviewUrl : activeLayer === 'native' ? nativeNdviUrl : srNdviUrl;

  const downloads = [
    { name: 'SR NDVI GeoTIFF (4x)', path: 'application/crop/ndvi_sr.tif', desc: '4x super-resolved NDVI raster' },
    { name: 'Native NDVI GeoTIFF', path: 'application/crop/ndvi_native.tif', desc: '10m resolution NDVI raster' },
    { name: 'NDVI Difference Map', path: 'application/crop/ndvi_difference.tif', desc: 'Pixel-level difference (SR - Native)' },
    { name: 'Uncertainty Map (GeoTIFF)', path: 'uncertainty/uncertainty_map.tif', desc: 'Stochastic diffusion variance' },
    { name: 'HTML Crop Report', path: 'report/report.html', desc: 'Standalone interactive HTML report' },
    { name: 'Markdown Report', path: 'report/report.md', desc: 'Human-readable documentation' },
    { name: 'JSON Manifest', path: 'manifest.json', desc: 'Digital asset inventory' },
  ];

  return (
    <div className="min-h-screen pb-16 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Top Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <Sprout className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white">Crop & Vegetation Monitoring</h1>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                Native-vs-SR Consistency
              </span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5 font-mono">Job: {jobId} · Sentinel-2 Red (B04) & NIR (B08)</div>
          </div>
        </div>
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Analyze Another Area
        </button>
      </header>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-8 items-start">
        {/* Left Column: Visualizers */}
        <div className="space-y-6">
          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Layers className="w-4 h-4 text-emerald-400" />
                <span>Vegetation Index Visualization</span>
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
                  Layer View
                </button>
              </div>
            </div>

            {viewMode === 'slider' ? (
              <div className="space-y-3">
                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40">
                  <CompareSlider
                    leftSrc={originalRgbUrl}
                    rightSrc={srRgbUrl}
                    leftLabel="◀ INPUT · 10 m (native)"
                    rightLabel="SR · ~2.5 m (4×) ▶"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>Drag to see resolution sharpening — left is native 10 m, right is 4× SR</span>
                  <span>Switch to Layer View to compare NDVI maps</span>
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
              {cropData?.interpretations?.map((txt, idx) => (
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
                {cropData?.limitations?.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right Column: Statistics & Downloads */}
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
                  {nativeStats?.mean != null ? nativeStats.mean.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">σ = {nativeStats?.std?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">SR Mean NDVI</div>
                <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
                  {srStats?.mean != null ? srStats.mean.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">σ = {srStats?.std?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">MAE (Consistency)</div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  {consistency?.mae != null ? consistency.mae.toFixed(4) : '—'}
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">Mean absolute diff</div>
              </div>
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">RMSE</div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  {consistency?.rmse != null ? consistency.rmse.toFixed(4) : '—'}
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
                    {((canopy?.low_or_potential_stress_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-amber-400 rounded-full"
                    style={{ width: `${(canopy?.low_or_potential_stress_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Moderate Vegetation (0.2 – 0.5)</span>
                  <span className="font-mono text-emerald-400">
                    {((canopy?.moderate_vegetation_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${(canopy?.moderate_vegetation_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Dense Canopy (&ge; 0.5)</span>
                  <span className="font-mono text-emerald-300">
                    {((canopy?.dense_canopy_fraction || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-emerald-400 rounded-full"
                    style={{ width: `${(canopy?.dense_canopy_fraction || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Downloads Card */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Generated Agricultural Artifacts</span>
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
                  <Download className="w-4 h-4 text-emerald-400 shrink-0" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
