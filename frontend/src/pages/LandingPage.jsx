import React from 'react';
import {
  Sprout,
  Building2,
  Flame,
  Zap,
  ArrowRight,
  ShieldCheck,
  Globe2,
  Sparkles,
  Layers,
  Activity,
  Cpu
} from 'lucide-react';

const APPLICATIONS = [
  {
    id: 'crop',
    title: 'Crop Monitoring',
    category: 'Agricultural Intelligence',
    desc: 'Multispectral B04 (Red) & B08 (NIR) vegetation index analysis, sub-field crop stress mapping, and canopy vigor quantification.',
    badge: '4× NDVI Representation',
    icon: Sprout,
    color: '#10b981',
    gradient: 'from-emerald-500/20 via-sky-500/10 to-transparent',
    border: 'border-emerald-500/30 hover:border-emerald-400',
    btnBg: 'bg-emerald-600 hover:bg-emerald-500',
    features: ['Native vs 4× SR NDVI', 'Canopy Stress Fractions', 'Radiometric Consistency'],
  },
  {
    id: 'urban',
    title: 'Urban Analysis',
    category: 'Infrastructure & Land Cover',
    desc: 'ESA WorldCover 7-class deep segmentation, built-up footprint extraction, urban sprawl density, and building cluster resolution.',
    badge: '7-Class Segmentation',
    icon: Building2,
    color: '#f59e0b',
    gradient: 'from-amber-500/20 via-sky-500/10 to-transparent',
    border: 'border-amber-500/30 hover:border-amber-400',
    btnBg: 'bg-amber-600 hover:bg-amber-500',
    features: ['WorldCover 7 Classes', 'Built-up Area Fractions', 'Boundary Delineation'],
  },
  {
    id: 'disaster',
    title: 'Disaster Management',
    category: 'Emergency & Resilience',
    desc: 'Bitemporal Sentinel-2 acquisition matching, automated CRS alignment, spectral change vector tracking, and damage zone reliability.',
    badge: 'Dual-Scene Bitemporal',
    icon: Flame,
    color: '#f43f5e',
    gradient: 'from-rose-500/20 via-amber-500/10 to-transparent',
    border: 'border-rose-500/30 hover:border-rose-400',
    btnBg: 'bg-rose-600 hover:bg-rose-500',
    features: ['Pre/Post Event Alignment', 'Spectral Change Vectors', 'Uncertainty Reliability Map'],
  },
];

export default function LandingPage({
  onSelectApplication,
  health,
}) {
  return (
    <main
      className="min-h-screen px-5 py-8 md:px-12 flex flex-col justify-between"
      style={{
        background:
          'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(6,182,212,0.12), transparent 70%), #050a14',
      }}
    >
      {/* Top Navigation Header */}
      <header className="max-w-6xl w-full mx-auto flex items-center justify-between pb-6 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/20"
            style={{ background: 'linear-gradient(135deg,#0891b2,#1d4ed8)' }}
          >
            <Globe2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-extrabold text-white text-lg tracking-tight flex items-center gap-2">
              PIXELSIGHT
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase font-mono tracking-widest">
                Copernicus Integrated
              </span>
            </div>
            <div className="text-[11px] text-slate-400 font-medium tracking-wide">
              Sentinel-2 Satellite Intelligence Platform
            </div>
          </div>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              health?.status === 'ok'
                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                : health?.status === 'offline'
                ? 'bg-red-400'
                : 'bg-amber-400 animate-pulse'
            }`}
          />
          <span className="text-slate-400 font-medium">
            {health?.status === 'ok' ? 'Copernicus & GPU Ready' : 'Connecting to Core API...'}
          </span>
        </div>
      </header>

      {/* Main Hero & Selection Section */}
      <div className="max-w-6xl w-full mx-auto my-12 space-y-12">
        {/* Title Banner */}
        <div className="text-center space-y-4 max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-bold tracking-widest uppercase">
            <Sparkles className="w-3.5 h-3.5" /> Next-Generation Earth Observation
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-black text-white tracking-tight leading-[1.08]">
            Choose what you want <br className="hidden sm:block" />
            <span className="bg-gradient-to-r from-sky-400 via-cyan-300 to-emerald-400 bg-clip-text text-transparent">
              to analyze
            </span>
          </h1>
          <p className="text-slate-400 text-sm md:text-base leading-relaxed">
            Direct integration with the Copernicus Data Space Ecosystem (CDSE). Select an AOI on the interactive map or upload multispectral Sentinel-2 GeoTIFF data for 4× super-resolution representation (~2.5 m equivalent).
          </p>
        </div>

        {/* 3 Primary Application Cards (Rule 3) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {APPLICATIONS.map((app) => {
            const Icon = app.icon;
            return (
              <div
                key={app.id}
                className={`group relative rounded-3xl p-6 bg-gradient-to-b ${app.gradient} bg-slate-900/90 border ${app.border} shadow-2xl transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div
                      className="w-12 h-12 rounded-2xl flex items-center justify-center shadow-md transition-transform group-hover:scale-110"
                      style={{
                        backgroundColor: `${app.color}20`,
                        border: `1px solid ${app.color}40`,
                        color: app.color,
                      }}
                    >
                      <Icon className="w-6 h-6" />
                    </div>
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2.5 py-1 rounded-full bg-slate-800/90 border border-slate-700/60 text-slate-300">
                      {app.badge}
                    </span>
                  </div>

                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                    {app.category}
                  </div>
                  <h2 className="text-xl font-black text-white mb-2 group-hover:text-cyan-200 transition-colors">
                    {app.title}
                  </h2>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    {app.desc}
                  </p>

                  {/* Feature Pills */}
                  <div className="space-y-1.5 mb-8">
                    {app.features.map((feat, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                        <span>{feat}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Primary Launch Action */}
                <button
                  onClick={() => onSelectApplication(app.id)}
                  className={`w-full py-3.5 px-4 rounded-xl ${app.btnBg} text-white font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-cyan-500/20 active:scale-95`}
                >
                  <span>Launch {app.title}</span>
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </button>
              </div>
            );
          })}
        </div>

        {/* Secondary / Advanced Section: Research Engine Studio */}
        <div className="p-6 rounded-3xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-colors flex flex-wrap items-center justify-between gap-6">
          <div className="flex items-center gap-4 max-w-xl">
            <div className="p-3 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 flex-shrink-0">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-bold text-white">Advanced Research & Model Exploration</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase font-mono">
                  100 Steps Diffusion
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Full research studio for benchmark validation, Fourier Ring Correlation (FRC), Modulation Transfer Function (MTF), and pixel-level stochastic variance mapping.
              </p>
            </div>
          </div>

          <button
            onClick={() => onSelectApplication('research')}
            className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white border border-slate-700 font-bold text-xs uppercase tracking-wider transition-all flex items-center gap-2"
          >
            <span>Open Research Studio</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Footer */}
      <footer className="max-w-6xl w-full mx-auto pt-6 border-t border-white/[0.06] text-center text-xs text-slate-500">
        PixelSight Satellite Intelligence Platform · Powered by Copernicus Data Space Ecosystem & ESA OpenSR LDSR-S2
      </footer>
    </main>
  );
}
