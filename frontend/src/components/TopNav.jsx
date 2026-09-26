import React from 'react';
import {
  Globe2, Sprout, Building2, Flame, Layers, FileStack,
  BarChart3, Cpu, Map, CheckCircle2
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'landing', label: 'Home', icon: Globe2 },
  { id: 'research', label: 'Core Engine', icon: Layers },
  { id: 'crop', label: 'Crop Monitoring', icon: Sprout },
  { id: 'urban', label: 'Urban Analysis', icon: Building2 },
  { id: 'disaster', label: 'Disaster Management', icon: Flame },
  { id: 'batch', label: 'Batch Processing', icon: FileStack },
  { id: 'classification', label: 'Classification', icon: Map },
  { id: 'ground_truth', label: 'Ground Truth', icon: CheckCircle2 },
  { id: 'evaluation', label: 'Evaluation / Research', icon: BarChart3 },
  { id: 'compute', label: 'Compute', icon: Cpu },
];

export default function TopNav({ activeView, onNavigate, health }) {
  return (
    <nav className="w-full bg-slate-950/90 border-b border-white/[0.08] backdrop-blur-md sticky top-0 z-40 px-4 md:px-8 py-2.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand */}
        <div
          onClick={() => onNavigate('landing')}
          className="flex items-center gap-2.5 cursor-pointer group shrink-0"
        >
          <div className="w-8 h-8 rounded-xl flex items-center justify-center bg-gradient-to-tr from-cyan-600 to-blue-600 shadow-md shadow-cyan-500/20 group-hover:scale-105 transition-transform">
            <Globe2 className="w-4 h-4 text-white" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-white text-sm tracking-tight">PIXELSIGHT</span>
            <span className="hidden sm:inline text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase font-mono">
              India Scope
            </span>
          </div>
        </div>

        {/* Scrollable Nav Items */}
        <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        {/* Backend health status badge */}
        <div className="hidden lg:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px] shrink-0">
          <span
            className={`w-2 h-2 rounded-full ${
              health?.status === 'ok' ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : 'bg-red-400'
            }`}
          />
          <span className="text-slate-400 font-mono">
            {health?.status === 'ok' ? 'API Online' : 'Connecting...'}
          </span>
        </div>
      </div>
    </nav>
  );
}
