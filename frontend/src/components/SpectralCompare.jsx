import React, { useState } from 'react';
import { Sprout, Waves, BarChart2 } from 'lucide-react';

export default function SpectralCompare({ data }) {
  const [activeTab, setActiveTab] = useState('ndvi'); // 'ndvi' or 'ndwi'

  if (!data) return null;

  const isNdvi = activeTab === 'ndvi';
  const lrMap = isNdvi ? data.ndvi_lr_b64 : data.ndwi_lr_b64;
  const srMap = isNdvi ? data.ndvi_sr_b64 : data.ndwi_sr_b64;

  const meanLr = isNdvi ? data.spectral?.mean_ndvi_lr : data.spectral?.mean_ndwi_lr;
  const meanSr = isNdvi ? data.spectral?.mean_ndvi_sr : data.spectral?.mean_ndwi_sr;
  const delta = isNdvi ? data.spectral?.ndvi_delta : data.spectral?.ndwi_delta;

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-emerald-400" />
            Spectral Consistency Verification (NDVI & NDWI)
          </h3>
          <p className="text-xs text-slate-400">
            Verifies that super-resolution enhancement preserves radiometrically accurate spectral signatures
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setActiveTab('ndvi')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'ndvi' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Sprout className="w-3.5 h-3.5" />
            NDVI (Vegetation)
          </button>
          <button
            onClick={() => setActiveTab('ndwi')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'ndwi' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Waves className="w-3.5 h-3.5" />
            NDWI (Water Bodies)
          </button>
        </div>
      </div>

      {/* Spectral Metrics Summary Row */}
      <div className="grid grid-cols-3 gap-3 mb-4 text-xs font-mono">
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Mean LR {isNdvi ? 'NDVI' : 'NDWI'}</span>
          <span className="text-slate-200 text-base font-bold">{meanLr?.toFixed(4) ?? '0.0000'}</span>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Mean SR {isNdvi ? 'NDVI' : 'NDWI'}</span>
          <span className="text-emerald-400 text-base font-bold">{meanSr?.toFixed(4) ?? '0.0000'}</span>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Spectral Shift Delta</span>
          <span className={`text-base font-bold ${delta >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {delta >= 0 ? `+${delta?.toFixed(4)}` : delta?.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Side-by-Side Spectral Index Maps */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* LR Spectral Map */}
        <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <div className="text-xs font-semibold text-slate-300 mb-2">
            LR Bicubic {isNdvi ? 'NDVI' : 'NDWI'} Heatmap
          </div>
          <div className="relative aspect-square w-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
            <img src={lrMap} alt="LR Spectral Index" className="w-full h-full object-contain" />
          </div>
        </div>

        {/* SR Spectral Map */}
        <div className="bg-slate-900/60 rounded-xl p-3 border border-emerald-500/30">
          <div className="text-xs font-semibold text-emerald-300 mb-2">
            Super-Resolved (SR) {isNdvi ? 'NDVI' : 'NDWI'} Heatmap
          </div>
          <div className="relative aspect-square w-full bg-slate-950 rounded-lg overflow-hidden border border-emerald-500/40 flex items-center justify-center shadow-lg shadow-emerald-950/30">
            <img src={srMap} alt="SR Spectral Index" className="w-full h-full object-contain" />
          </div>
        </div>

      </div>

    </div>
  );
}
