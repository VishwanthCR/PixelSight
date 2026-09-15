import React, { useState } from 'react';
import { Eye, EyeOff, Sparkles, Image as ImageIcon } from 'lucide-react';

export default function ResultViewer({ data }) {
  const [bandMode, setBandMode] = useState('rgb'); // 'rgb' or 'false_color'

  if (!data) return null;

  const lrSrc = bandMode === 'rgb' ? data.lr_image_b64 : data.lr_false_color_b64;
  const srSrc = bandMode === 'rgb' ? data.sr_image_b64 : data.sr_false_color_b64;

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            Super-Resolution Mapping Comparison
          </h3>
          <p className="text-xs text-slate-400">
            Original 10m Sentinel-2 Input vs Neural Super-Resolved Enhancement
          </p>
        </div>

        {/* RGB vs False Color Toggle */}
        <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setBandMode('rgb')}
            className={`px-3 py-1 rounded-lg transition-all ${
              bandMode === 'rgb' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            True Color (RGB)
          </button>
          <button
            onClick={() => setBandMode('false_color')}
            className={`px-3 py-1 rounded-lg transition-all ${
              bandMode === 'false_color' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            False Color (NIR, R, G)
          </button>
        </div>
      </div>

      {/* Side-by-Side Image Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Original LR */}
        <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <ImageIcon className="w-4 h-4 text-slate-400" />
              Original LR Input Patch (10m Spatial Resolution)
            </span>
            <span className="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded">
              Input: {data.lineage?.input_shape || '32x32'}
            </span>
          </div>
          <div className="relative aspect-square w-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
            <img
              src={lrSrc}
              alt="Low Resolution Sentinel-2"
              className="w-full h-full object-contain image-rendering-pixelated"
            />
          </div>
        </div>

        {/* Enhanced SR */}
        <div className="bg-slate-900/60 rounded-xl p-3 border border-blue-500/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-blue-300 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-blue-400" />
              Super-Resolved Output (Enhanced Spatial Scale)
            </span>
            <span className="text-[10px] font-mono bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/30">
              {data.lineage?.spatial_scale || '2x Enhanced'}
            </span>
          </div>
          <div className="relative aspect-square w-full bg-slate-950 rounded-lg overflow-hidden border border-blue-500/40 flex items-center justify-center shadow-lg shadow-blue-950/40">
            <img
              src={srSrc}
              alt="Super Resolved Sentinel-2"
              className="w-full h-full object-contain"
            />
          </div>
        </div>

      </div>

    </div>
  );
}
