import React from 'react';
import { Layers, Activity, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function Header({ health, isPlaceholderModel }) {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand & System Title */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-blue-600/20 border border-blue-500/40 rounded-xl text-blue-400">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white">PixelSight</h1>
              <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-full">
                Sentinel-2 SR Mapping
              </span>
            </div>
            <p className="text-xs text-slate-400">
              SIH Problem Statement 26142 | NTRO Satellite Intelligence Engine
            </p>
          </div>
        </div>

        {/* System Lineage & Status Badges */}
        <div className="flex items-center gap-3">
          {/* Checkpoint Status Indicator */}
          {isPlaceholderModel ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
              <AlertCircle className="w-4 h-4 text-amber-400 animate-pulse" />
              <span>Model Checkpoint: <strong>Scaffolded (Awaiting Teammate .pt)</strong></span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Model Checkpoint: <strong>Loaded & Active</strong></span>
            </div>
          )}

          {/* Backend Health Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 text-xs">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>FastAPI: <strong className="text-emerald-400">{health ? 'Online' : 'Connecting...'}</strong></span>
          </div>
        </div>

      </div>
    </header>
  );
}
