import React from 'react';
import { Cpu, Info } from 'lucide-react';

export default function ResidualMapView({ data }) {
  if (!data) return null;

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            High-Frequency Residual Map (SR − Bicubic Baseline)
          </h3>
          <p className="text-xs text-slate-400">
            Exposes exact structural and textural details introduced by the neural network beyond standard bicubic interpolation
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        
        {/* Residual Visual */}
        <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <div className="relative aspect-square w-full max-h-96 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
            <img
              src={data.residual_map_b64}
              alt="Residual Difference Map"
              className="w-full h-full object-contain"
            />
            {/* Viridis Color Legend */}
            <div className="absolute bottom-3 left-3 right-3 bg-slate-900/80 backdrop-blur-sm p-2 rounded-lg border border-slate-800 flex items-center justify-between text-[11px] font-mono">
              <span className="text-slate-400">Low Diff</span>
              <div className="h-2 flex-1 mx-3 rounded bg-gradient-to-r from-purple-900 via-teal-500 to-yellow-300"></div>
              <span className="text-cyan-300">High Detail Gain</span>
            </div>
          </div>
        </div>

        {/* Technical Explanation */}
        <div className="space-y-4 text-xs text-slate-300">
          <div className="bg-cyan-950/30 border border-cyan-500/30 rounded-xl p-4">
            <h4 className="font-semibold text-cyan-300 flex items-center gap-1.5 mb-2 text-sm">
              <Info className="w-4 h-4" />
              What Changed in this Patch?
            </h4>
            <p className="leading-relaxed text-slate-300">
              The residual map isolates the <strong>high-frequency spatial details</strong> (building boundaries, road edges, field lines) synthesized by the Super-Resolution model.
            </p>
          </div>

          <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 space-y-2">
            <div className="flex justify-between font-mono">
              <span className="text-slate-400">Spatial Resolution Multiplier:</span>
              <span className="text-cyan-300 font-bold">{data.lineage?.spatial_scale || '2x'}</span>
            </div>
            <div className="flex justify-between font-mono">
              <span className="text-slate-400">Input Resolution:</span>
              <span>10m Sentinel-2 GSD</span>
            </div>
            <div className="flex justify-between font-mono">
              <span className="text-slate-400">Enhanced Output GSD:</span>
              <span className="text-emerald-400 font-bold">5m Sub-pixel Resolution</span>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
