import React, { useState, useEffect } from 'react';
import {
  Cpu, HardDrive, Zap, ShieldCheck, ArrowLeft, RefreshCw,
  Activity, Layers, Sliders, CheckCircle2, AlertTriangle, Monitor
} from 'lucide-react';
import { fetchComputeProfile, fetchComputePlan } from '../api/srmApi.js';

export default function ComputeProfilePage({ onBack }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [inputSize, setInputSize] = useState(512);
  const [plan, setPlan] = useState(null);

  const loadProfile = () => {
    setLoading(true);
    fetchComputeProfile()
      .then(res => {
        setProfile(res);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    loadProfile();
  }, []);

  useEffect(() => {
    fetchComputePlan(inputSize, inputSize, 1, 1)
      .then(setPlan)
      .catch(() => {});
  }, [inputSize]);

  const hw = profile?.hardware || {};
  const primaryGpu = profile?.gpu?.primary || hw?.primary_gpu;
  const cpu = profile?.cpu || {};
  const cuda = profile?.cuda || {};

  return (
    <div className="min-h-screen pb-20 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-8">
        <div className="flex items-center gap-3.5">
          <button
            onClick={onBack}
            className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
            title="Back to Home"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Cpu className="w-4 h-4" />
              </div>
              <h1 className="text-xl font-bold text-white tracking-tight">Adaptive Compute Engine</h1>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300">
                {profile?.execution_environment || 'LOCAL_SINGLE_GPU'}
              </span>
            </div>
            <div className="text-xs text-slate-400 mt-1 font-medium">
              Runtime hardware discovery, VRAM memory guard, and dynamic tiling execution planner
            </div>
          </div>
        </div>

        <button
          onClick={loadProfile}
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh Detection
        </button>
      </header>

      {/* Main Grid */}
      <div className="space-y-8">
        {/* Core Hardware Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* GPU Card */}
          <div className="card p-5 border-l-4 border-l-cyan-500 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="uppercase font-semibold tracking-wider">GPU Accelerator</span>
              <Zap className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-lg font-bold text-white truncate">
              {primaryGpu ? primaryGpu.name : 'CPU Only (Fallback)'}
            </div>
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">VRAM Allocation</span>
              <span className="text-cyan-300 font-bold">
                {primaryGpu ? `${primaryGpu.vram_gb} GB` : '0.0 GB'}
              </span>
            </div>
          </div>

          {/* CUDA Card */}
          <div className="card p-5 border-l-4 border-l-emerald-500 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="uppercase font-semibold tracking-wider">CUDA Runtime</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-lg font-bold text-white">
              {cuda.available ? 'CUDA Available' : 'CPU Software Mode'}
            </div>
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Version</span>
              <span className="text-emerald-400 font-bold">{cuda.version || '12.0'}</span>
            </div>
          </div>

          {/* CPU Card */}
          <div className="card p-5 border-l-4 border-l-purple-500 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="uppercase font-semibold tracking-wider">Host CPU</span>
              <Monitor className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-lg font-bold text-white truncate" title={cpu.model || hw.cpu_model}>
              {cpu.model || hw.cpu_model || 'Detected CPU'}
            </div>
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Cores / Threads</span>
              <span className="text-purple-300 font-bold">
                {cpu.cores || hw.cpu_physical_cores || 8}C / {cpu.logical_processors || hw.cpu_logical_cores || 16}T
              </span>
            </div>
          </div>

          {/* RAM Card */}
          <div className="card p-5 border-l-4 border-l-amber-500 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="uppercase font-semibold tracking-wider">System Memory</span>
              <HardDrive className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-lg font-bold text-white">
              {cpu.ram_gb || hw.system_ram_gb ? `${cpu.ram_gb || hw.system_ram_gb} GB RAM` : '16 GB RAM'}
            </div>
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Execution Mode</span>
              <span className="text-amber-300 font-bold">{profile?.execution_backend || 'single_gpu'}</span>
            </div>
          </div>
        </div>

        {/* Dynamic Workload Planner Section */}
        <div className="card p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
            <div className="flex items-center gap-2">
              <Sliders className="w-5 h-5 text-cyan-400" />
              <h2 className="text-base font-bold text-white">Interactive Workload &amp; Tiling Execution Planner</h2>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              128×128 model tiles · 12 px overlap · 100 diffusion steps
            </span>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Target Scene Size (Square Dimensions):</span>
              <span className="font-mono text-cyan-300 font-bold text-sm">{inputSize} × {inputSize} pixels</span>
            </div>
            <input
              type="range"
              min="128"
              max="1024"
              step="64"
              value={inputSize}
              onChange={e => setInputSize(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>128×128 (1 tile)</span>
              <span>256×256 (4 tiles)</span>
              <span>512×512 (25 tiles)</span>
              <span>1024×1024 (81 tiles)</span>
            </div>
          </div>

          {plan && (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-4 border-t border-white/[0.06] text-xs">
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                <span className="text-[10px] uppercase text-slate-500 block mb-1">Backend</span>
                <span className="font-mono font-bold text-cyan-300">{plan.backend?.toUpperCase()}</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                <span className="text-[10px] uppercase text-slate-500 block mb-1">Parallelism</span>
                <span className="font-mono font-bold text-slate-200">{plan.parallelism}</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                <span className="text-[10px] uppercase text-slate-500 block mb-1">Workers</span>
                <span className="font-mono font-bold text-emerald-400">{plan.workers} Worker</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                <span className="text-[10px] uppercase text-slate-500 block mb-1">Estimated Tiles</span>
                <span className="font-mono font-bold text-amber-300">{plan.estimated_tiles_per_scene} Tiles</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                <span className="text-[10px] uppercase text-slate-500 block mb-1">Scale &amp; Steps</span>
                <span className="font-mono font-bold text-purple-300">{plan.scale_factor}× · {plan.diffusion_steps} steps</span>
              </div>
            </div>
          )}

          {plan?.reason && (
            <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-500/20 text-xs text-slate-300 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
              <span><strong>Planner Decision:</strong> {plan.reason}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
