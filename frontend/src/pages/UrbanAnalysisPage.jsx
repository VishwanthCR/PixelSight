import React, { useState } from 'react';
import {
  Building2, Download, FileText, RotateCcw, AlertTriangle,
  Info, Activity, Layers, Trees, Waves, Mountain,
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import { resultFileUrl } from '../api/srmApi.js';

const CLASS_COLORS = {
  tree: '#28b45a',
  shrubland: '#78aa50',
  grassland: '#aad264',
  cropland: '#dcbe46',
  built_up: '#d25a37',
  bare: '#96876e',
  water: '#327dd2',
};

export default function UrbanAnalysisPage({ job, results, report, onReset }) {
  const [activeLayer, setActiveLayer] = useState('sr_seg');
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'single'

  const jobId = job?.job_id || results?.job_id;
  const urbanData = results?.outputs?.urban || report?.urban_analysis || {};
  const indicators = urbanData?.indicators || {};
  const distribution = urbanData?.class_distribution || {};

  const srSegPreview    = resultFileUrl(jobId, 'application/urban/previews/segmentation_sr.png');
  const nativeSegPreview = resultFileUrl(jobId, 'application/urban/previews/segmentation_native.png');
  const builtupPreview   = resultFileUrl(jobId, 'application/urban/previews/builtup_preview.png');
  const diffPreview      = resultFileUrl(jobId, 'application/urban/previews/urban_difference.png');
  const originalRgbUrl   = resultFileUrl(jobId, 'input/original_preview.png');
  const srRgbUrl         = resultFileUrl(jobId, 'super_resolution/sr_preview.png');

  const currentPreview =
    activeLayer === 'builtup'
      ? builtupPreview
      : activeLayer === 'diff'
      ? diffPreview
      : activeLayer === 'native'
      ? nativeSegPreview
      : srSegPreview;

  const downloads = [
    { name: 'SR Segmentation GeoTIFF (4x)', path: 'application/urban/segmentation_sr.tif', desc: 'WorldCover 7-class raster' },
    { name: 'Native Segmentation GeoTIFF', path: 'application/urban/segmentation_native.tif', desc: 'Input-resolution classification' },
    { name: 'Built-up Mask GeoTIFF', path: 'application/urban/builtup.tif', desc: 'Binary building cluster footprint' },
    { name: 'Urban Difference GeoTIFF', path: 'application/urban/urban_difference.tif', desc: 'Reclassification discrepancy mask' },
    { name: 'HTML Urban Report', path: 'report/report.html', desc: 'Standalone interactive HTML report' },
    { name: 'JSON Manifest', path: 'manifest.json', desc: 'Full digital asset manifest' },
  ];

  return (
    <div className="min-h-screen pb-16 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-amber-500/10 border border-amber-500/30 text-amber-400">
            <Building2 className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white">Urban & Land-Cover Analysis</h1>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300">
                Model-Output Comparison
              </span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5 font-mono">Job: {jobId} · ESA WorldCover 7-class UNet</div>
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
                <Layers className="w-4 h-4 text-amber-400" />
                <span>Classified Surface Models</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={() => setViewMode('slider')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'slider' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Native vs SR Slider
                </button>
                <button
                  onClick={() => setViewMode('single')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    viewMode === 'single' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-white/[0.04] text-slate-400'
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
                    leftSrc={originalRgbUrl}
                    rightSrc={srRgbUrl}
                    leftLabel="◀ INPUT · 10 m (native)"
                    rightLabel="SR · ~2.5 m (4×) ▶"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                  <span>Drag to compare native vs super-resolved image quality</span>
                  <span>Switch to Layer Selector for segmentation maps</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setActiveLayer('native')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'native' ? 'bg-amber-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Native Seg
                  </button>
                  <button
                    onClick={() => setActiveLayer('sr_seg')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'sr_seg' ? 'bg-amber-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    SR Seg (4x)
                  </button>
                  <button
                    onClick={() => setActiveLayer('builtup')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'builtup' ? 'bg-amber-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Built-up Footprint
                  </button>
                  <button
                    onClick={() => setActiveLayer('diff')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      activeLayer === 'diff' ? 'bg-amber-500 text-slate-950' : 'bg-white/[0.04] text-slate-400'
                    }`}
                  >
                    Difference Mask
                  </button>
                </div>
                <div className="h-[420px] rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 flex items-center justify-center">
                  <img src={currentPreview} alt="Urban Layer" className="max-h-full max-w-full object-contain" />
                </div>
              </div>
            )}
          </div>

          {/* Scientific Interpretation Card */}
          <div className="card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Activity className="w-4 h-4 text-amber-400" />
              <span>Urban Planning & Structural Interpretation</span>
            </div>
            <div className="space-y-2 text-sm text-slate-300 leading-relaxed">
              {urbanData?.interpretations?.map((txt, idx) => (
                <div key={idx} className="flex items-start gap-2.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-2 shrink-0" />
                  <span>{txt}</span>
                </div>
              ))}
            </div>

            {/* Limitations Callout */}
            <div className="rounded-xl p-4 bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-300">
                <AlertTriangle className="w-4 h-4" />
                <span>Segmentation Research Notice</span>
              </div>
              <ul className="list-disc pl-5 space-y-1 text-slate-400">
                {urbanData?.limitations?.map((lim, idx) => (
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
            <span className="text-sm font-semibold text-white">Land-Cover Physical Indicators</span>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Building2 className="w-3.5 h-3.5 text-amber-400" /> Built-up Area
                </div>
                <div className="text-xl font-bold font-mono text-amber-300 mt-1">
                  {((indicators?.built_up_fraction || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">
                  {indicators?.connected_built_up_regions ?? 0} connected clusters
                </div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Trees className="w-3.5 h-3.5 text-emerald-400" /> Vegetation
                </div>
                <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
                  {((indicators?.vegetation_fraction || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">
                  {indicators?.connected_tree_regions ?? 0} tree clusters
                </div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Waves className="w-3.5 h-3.5 text-sky-400" /> Water Body
                </div>
                <div className="text-xl font-bold font-mono text-sky-300 mt-1">
                  {((indicators?.water_fraction || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">{indicators?.water_area_pixels ?? 0} px</div>
              </div>

              <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.05]">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Mountain className="w-3.5 h-3.5 text-stone-400" /> Bare Surface
                </div>
                <div className="text-xl font-bold font-mono text-stone-300 mt-1">
                  {((indicators?.bare_fraction || 0) * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-600 mt-0.5">{indicators?.bare_area_pixels ?? 0} px</div>
              </div>
            </div>
          </div>

          {/* Class Distribution Table */}
          <div className="card p-5 space-y-3">
            <span className="text-sm font-semibold text-white">Full Class Distribution</span>
            <div className="space-y-2">
              {Object.entries(distribution).map(([key, data]) => (
                <div key={key} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="flex items-center gap-2 text-slate-300">
                      <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: CLASS_COLORS[key] || '#94a3b8' }} />
                      {data.label}
                    </span>
                    <span className="font-mono text-slate-400">{data.percent?.toFixed(1) ?? '—'}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${data.percent || 0}%`,
                        backgroundColor: CLASS_COLORS[key] || '#94a3b8',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

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
                  <Download className="w-4 h-4 text-amber-400 shrink-0" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
