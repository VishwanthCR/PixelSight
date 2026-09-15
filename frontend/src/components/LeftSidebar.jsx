/* PixelSight — Left Sidebar: Source Imagery & Pipeline */

import React, { useRef, useState } from 'react';
import {
  Upload, Play, CheckCircle2, Loader2, Circle,
  RefreshCw, Cpu, Layers, ChevronDown
} from 'lucide-react';

const USE_CASES = [
  { id: 'urban', label: 'Urban', color: '#f59e0b' },
  { id: 'crop', label: 'Agriculture', color: '#10b981' },
  { id: 'disaster', label: 'Disaster / Coastal', color: '#ef4444' },
];

const PIPELINE_STEPS = [
  { id: 'preprocessing', label: 'Preprocessing', icon: Layers },
  { id: 'inference',     label: 'Inference (SR)',  icon: Cpu   },
  { id: 'analysis',      label: 'Analysis',        icon: RefreshCw },
  { id: 'complete',      label: 'Complete',         icon: CheckCircle2 },
];

export default function LeftSidebar({ onRunAnalysis, loading, pipelineStep, uploadedFile, onFileChange }) {
  const fileRef = useRef();
  const [useCase, setUseCase] = useState('urban');
  const [nSamples, setNSamples] = useState(10);

  /* Resolve step index (0-based) from pipelineStep string */
  const stepIndex = PIPELINE_STEPS.findIndex(s => s.id === pipelineStep);
  const overallPct = pipelineStep === 'complete' ? 100
    : pipelineStep === 'analysis'     ? 75
    : pipelineStep === 'inference'    ? 50
    : pipelineStep === 'preprocessing'? 20
    : 0;

  function handleFilePick(e) {
    const f = e.target.files?.[0];
    if (f) onFileChange(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) onFileChange(f);
  }

  function handleRun() {
    onRunAnalysis({ useCase, nSamples, regionId: 'region2', patchIndex: 0, file: uploadedFile });
  }

  return (
    <aside className="flex flex-col h-full overflow-y-auto" style={{ width: 272, minWidth: 272 }}>

      {/* ── Logo / Brand ── */}
      <div className="px-5 py-4 border-b border-white/5 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#06b6d4" strokeWidth="2" strokeLinejoin="round"/>
            <path d="M2 17l10 5 10-5" stroke="#06b6d4" strokeWidth="2" strokeLinejoin="round"/>
            <path d="M2 12l10 5 10-5" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round"/>
          </svg>
        </div>
        <div>
          <div className="text-sm font-bold text-white tracking-tight">PixelSight</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">SR Mapping System</div>
        </div>
      </div>

      <div className="flex-1 p-4 space-y-4">

        {/* ── Source Imagery Upload ── */}
        <div>
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Source Imagery</div>

          {/* Dropzone */}
          <div
            onDrop={handleDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 rounded-xl p-4 cursor-pointer transition-all group text-center"
            style={{ background: uploadedFile ? 'rgba(6,182,212,0.04)' : 'rgba(255,255,255,0.02)' }}
          >
            {uploadedFile ? (
              <>
                <div className="w-full h-24 rounded-lg overflow-hidden mb-2 bg-slate-800 flex items-center justify-center">
                  {uploadedFile.type?.startsWith('image/') ? (
                    <img src={URL.createObjectURL(uploadedFile)} alt="preview" className="w-full h-full object-cover rounded-lg" />
                  ) : (
                    <div className="flex flex-col items-center gap-1 text-slate-400">
                      <Layers className="w-8 h-8 opacity-50" />
                      <span className="text-xs">.npy patch</span>
                    </div>
                  )}
                </div>
                <p className="text-xs text-slate-400 truncate">{uploadedFile.name}</p>
                <p className="text-[10px] text-slate-600 mt-0.5">{(uploadedFile.size / 1024).toFixed(1)} KB</p>
              </>
            ) : (
              <>
                <Upload className="w-6 h-6 text-slate-600 group-hover:text-cyan-400 mx-auto mb-2 transition-colors" />
                <p className="text-xs text-slate-400 group-hover:text-slate-300 transition-colors">
                  Drop Sentinel-2 patch here
                </p>
                <p className="text-[10px] text-slate-600 mt-1">.npy · .tif · .png · .jpg</p>
              </>
            )}
            <input ref={fileRef} type="file" accept=".npy,.tif,.tiff,.png,.jpg" onChange={handleFilePick} className="hidden" />
          </div>

          {/* Use-Case Selector */}
          <div className="mt-3">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Analysis Lens</div>
            <div className="flex gap-1.5">
              {USE_CASES.map(uc => (
                <button
                  key={uc.id}
                  onClick={() => setUseCase(uc.id)}
                  className="flex-1 text-[10px] py-1.5 rounded-lg font-semibold transition-all border"
                  style={{
                    borderColor: useCase === uc.id ? uc.color + '66' : 'rgba(255,255,255,0.07)',
                    background: useCase === uc.id ? uc.color + '18' : 'transparent',
                    color: useCase === uc.id ? uc.color : '#64748b',
                  }}
                >
                  {uc.label}
                </button>
              ))}
            </div>
          </div>

          {/* MC Passes */}
          <div className="mt-3 flex items-center justify-between">
            <span className="text-[10px] text-slate-500">MC Dropout Passes</span>
            <div className="flex items-center gap-2">
              <input
                type="range" min="5" max="30" value={nSamples}
                onChange={e => setNSamples(+e.target.value)}
                className="w-20 accent-cyan-500"
              />
              <span className="text-[10px] font-mono text-cyan-400 w-4">{nSamples}</span>
            </div>
          </div>
        </div>

        {/* ── Run Analysis Button ── */}
        <button
          onClick={handleRun}
          disabled={loading}
          className="w-full py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
          style={{
            background: loading ? '#1e3a5f' : 'linear-gradient(135deg, #0891b2 0%, #2563eb 100%)',
            color: loading ? '#64748b' : 'white',
            boxShadow: loading ? 'none' : '0 0 20px rgba(6,182,212,0.35)',
          }}
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" />Processing...</>
          ) : (
            <><Play className="w-4 h-4 fill-white" />Run SR Analysis</>
          )}
        </button>

        {/* ── Analysis Progress Pipeline ── */}
        <div>
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Analysis Progress</div>

          {/* Overall bar */}
          <div className="mb-4">
            <div className="flex justify-between text-[10px] mb-1.5">
              <span className="text-slate-500">Overall progress</span>
              <span className="font-mono text-cyan-400">{overallPct}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={overallPct > 0 && overallPct < 100 ? 'progress-bar-active h-full rounded-full' : 'h-full rounded-full transition-all duration-700'}
                style={{
                  width: `${overallPct}%`,
                  background: overallPct === 100 ? '#10b981' : overallPct === 0 ? 'transparent' : undefined,
                }}
              />
            </div>
          </div>

          {/* Step items */}
          <div className="space-y-2">
            {PIPELINE_STEPS.map((step, i) => {
              const isDone   = pipelineStep === 'complete' || (stepIndex > i);
              const isActive = stepIndex === i && pipelineStep !== 'complete';
              const Icon     = step.icon;
              return (
                <div key={step.id} className="flex items-center gap-3 py-1">
                  <div className={`w-5 h-5 rounded-full border flex items-center justify-center flex-shrink-0 ${
                    isDone   ? 'border-emerald-500/40 bg-emerald-500/10' :
                    isActive ? 'border-cyan-500/40 bg-cyan-500/10' :
                               'border-slate-700 bg-transparent'
                  }`}>
                    {isDone   ? <CheckCircle2 className="w-3 h-3 text-emerald-400" /> :
                     isActive ? <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" /> :
                                <Circle className="w-3 h-3 text-slate-700" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`text-xs font-medium ${isDone ? 'text-emerald-400' : isActive ? 'text-cyan-300' : 'text-slate-600'}`}>
                      {step.label}
                    </div>
                  </div>
                  {isDone && <CheckCircle2 className="w-4 h-4 text-emerald-500/60 flex-shrink-0" />}
                  {isActive && <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse flex-shrink-0" />}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Sample Regions (if no file uploaded) ── */}
        {!uploadedFile && (
          <div>
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Demo Patches</div>
            <div className="space-y-1.5">
              {[
                { id: 'region2', label: 'Urban — Chennai', tag: 'Urban' },
                { id: 'region1', label: 'Agriculture Sector', tag: 'Crop' },
                { id: 'region3', label: 'Coastal / Flood Zone', tag: 'Disaster' },
              ].map(r => (
                <button
                  key={r.id}
                  onClick={() => onRunAnalysis({ useCase: r.id === 'region2' ? 'urban' : r.id === 'region1' ? 'crop' : 'disaster', nSamples, regionId: r.id, patchIndex: 0 })}
                  disabled={loading}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-slate-800 hover:border-slate-700 hover:bg-white/[0.02] transition-all disabled:opacity-40"
                >
                  <span className="text-xs text-slate-300">{r.label}</span>
                  <span className="badge" style={{ background: 'rgba(6,182,212,0.1)', color: '#06b6d4', border: '1px solid rgba(6,182,212,0.2)' }}>{r.tag}</span>
                </button>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* Footer badge */}
      <div className="px-4 pb-4">
        <div className="text-[9px] text-slate-700 text-center uppercase tracking-widest">
          SIH 26142 · NTRO · Sentinel-2 10m
        </div>
      </div>

    </aside>
  );
}
