import React, { useState } from 'react';
import { ShieldAlert, HelpCircle, Activity } from 'lucide-react';

export default function UncertaintyOverlay({ data }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!data) return null;

  const meanUnc = data.metrics?.mean_uncertainty ?? 0.0;
  const maxUnc = data.metrics?.max_uncertainty ?? 0.0;
  const confidenceScore = Math.max(0, Math.min(100, (1 - meanUnc * 3) * 100)).toFixed(1);

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            Monte Carlo Dropout Uncertainty Quantification (std_dev)
          </h3>
          <p className="text-xs text-slate-400">
            Pixel-level variance heatmap computed across forward stochastic dropout sampling passes
          </p>
        </div>

        {/* Toggle Overlay */}
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            showHeatmap ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-slate-800 text-slate-400'
          }`}
        >
          {showHeatmap ? 'Heatmap: Visible' : 'Heatmap: Hidden'}
        </button>
      </div>

      {/* Grid: Image + Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        
        {/* Heatmap Card */}
        <div className="md:col-span-2 bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <div className="relative aspect-square w-full max-h-96 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
            {showHeatmap ? (
              <img
                src={data.uncertainty_map_b64}
                alt="Per-pixel Uncertainty Heatmap"
                className="w-full h-full object-contain"
              />
            ) : (
              <img
                src={data.sr_image_b64}
                alt="SR Output"
                className="w-full h-full object-contain"
              />
            )}

            {/* Infernal Colorbar Legend */}
            <div className="absolute bottom-3 left-3 right-3 bg-slate-900/80 backdrop-blur-sm p-2 rounded-lg border border-slate-800 flex items-center justify-between text-[11px] font-mono">
              <span className="text-emerald-400">0.0 (High Confidence)</span>
              <div className="h-2 flex-1 mx-3 rounded bg-gradient-to-r from-black via-purple-600 via-orange-500 to-yellow-300"></div>
              <span className="text-amber-400">High Variance (std_dev)</span>
            </div>
          </div>
        </div>

        {/* Confidence Stats Cards */}
        <div className="space-y-4">
          
          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
            <div className="text-xs text-slate-400 flex items-center justify-between mb-1">
              <span>Overall Model Confidence</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400">{confidenceScore}%</div>
            <p className="text-[11px] text-slate-400 mt-1">
              Derived from per-pixel std_dev across Monte Carlo passes
            </p>
          </div>

          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
            <div className="text-xs text-slate-400 mb-1">Mean Standard Deviation</div>
            <div className="text-xl font-bold font-mono text-amber-300">{meanUnc.toFixed(4)}</div>
            <p className="text-[11px] text-slate-500 mt-1">Average pixel variance</p>
          </div>

          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
            <div className="text-xs text-slate-400 mb-1">Max Standard Deviation</div>
            <div className="text-xl font-bold font-mono text-orange-400">{maxUnc.toFixed(4)}</div>
            <p className="text-[11px] text-slate-500 mt-1">Peak edge uncertainty</p>
          </div>

        </div>

      </div>

    </div>
  );
}
