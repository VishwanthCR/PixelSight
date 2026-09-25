import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers, Upload, Play, RotateCcw, CheckCircle2, XCircle,
  Clock, Zap, AlertTriangle, ChevronDown, ChevronUp,
  FileStack, BarChart3, Cpu, Download, ExternalLink,
  Sparkles, ArrowLeft
} from 'lucide-react';
import { startBatchProcessing, listBatchJobs, getJob, fetchComputeProfile, resultFileUrl } from '../api/srmApi.js';

const APP_OPTIONS = [
  { id: 'research', label: 'Research / LDSR-S2', color: '#06b6d4', desc: '100-step diffusion SR + uncertainty + segmentation' },
  { id: 'crop',    label: 'Crop Monitoring',    color: '#10b981', desc: 'NDVI analysis, photosynthetic canopy assessment' },
  { id: 'urban',   label: 'Urban Analysis',     color: '#f59e0b', desc: '7-class WorldCover segmentation, built-up footprint' },
];

function StatusPill({ status }) {
  const map = {
    queued:     { bg: 'bg-slate-800',        text: 'text-slate-400',  icon: <Clock className="w-3 h-3" /> },
    processing: { bg: 'bg-sky-900/50',       text: 'text-sky-300',   icon: <div className="w-3 h-3 border border-sky-400 border-t-transparent rounded-full animate-spin" /> },
    completed:  { bg: 'bg-emerald-900/40',   text: 'text-emerald-400', icon: <CheckCircle2 className="w-3 h-3" /> },
    failed:     { bg: 'bg-red-900/40',       text: 'text-red-400',   icon: <XCircle className="w-3 h-3" /> },
  };
  const s = map[status] || map.queued;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider ${s.bg} ${s.text}`}>
      {s.icon}{status}
    </span>
  );
}

function ProgressBar({ progress }) {
  return (
    <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden w-full">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{
          width: `${progress || 0}%`,
          background: 'linear-gradient(90deg, #0891b2, #38bdf8)',
          boxShadow: '0 0 8px rgba(56,189,248,0.4)',
        }}
      />
    </div>
  );
}

function JobCard({ job, jobId, filename }) {
  const data = job || { status: 'queued', stage: 'queued', progress: 0 };
  const isComplete = data.status === 'completed';
  const isFailed = data.status === 'failed';
  const srPreview = isComplete ? resultFileUrl(jobId, 'super_resolution/sr_preview.png') : null;

  return (
    <div className={`rounded-xl border p-4 transition-all duration-300 ${
      isComplete ? 'border-emerald-500/30 bg-emerald-500/[0.04]' :
      isFailed   ? 'border-red-500/30 bg-red-500/[0.04]' :
                   'border-slate-700/60 bg-slate-900/40'
    }`}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold text-slate-300 truncate">{filename}</div>
          <div className="font-mono text-[10px] text-slate-500 mt-0.5">{jobId}</div>
        </div>
        <StatusPill status={data.status} />
      </div>

      <ProgressBar progress={data.progress} />

      <div className="flex items-center justify-between mt-2">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">{data.stage}</span>
        <span className="text-[10px] font-mono text-slate-500">{(data.progress || 0).toFixed(0)}%</span>
      </div>

      {isFailed && data.error && (
        <div className="mt-2 p-2 rounded-lg bg-red-900/20 border border-red-500/20 text-xs text-red-300">
          {data.error}
        </div>
      )}

      {isComplete && srPreview && (
        <div className="mt-3 rounded-lg overflow-hidden border border-emerald-500/20" style={{ height: 80 }}>
          <img src={srPreview} alt="SR preview" className="w-full h-full object-cover" />
        </div>
      )}
    </div>
  );
}

export default function BatchProcessingPage({ onBack }) {
  const [selectedApp, setSelectedApp] = useState('research');
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [jobStates, setJobStates] = useState({});  // { jobId: jobData }
  const [computeProfile, setComputeProfile] = useState(null);
  const [showProfile, setShowProfile] = useState(false);
  const fileInputRef = useRef(null);
  const pollingRef = useRef(null);

  // Fetch compute profile on mount
  useEffect(() => {
    fetchComputeProfile().then(setComputeProfile).catch(() => {});
  }, []);

  // Poll job states while batch is running
  useEffect(() => {
    if (!batchResult?.jobs?.length) return;
    const allDone = () => Object.values(jobStates).every(j =>
      j.status === 'completed' || j.status === 'failed'
    );

    if (allDone() && Object.keys(jobStates).length === batchResult.jobs.length) {
      clearInterval(pollingRef.current);
      return;
    }

    pollingRef.current = setInterval(async () => {
      const updates = {};
      await Promise.all(
        batchResult.jobs.map(async ({ job_id }) => {
          try {
            const data = await getJob(job_id);
            updates[job_id] = data;
          } catch {}
        })
      );
      setJobStates(prev => ({ ...prev, ...updates }));
    }, 1500);

    return () => clearInterval(pollingRef.current);
  }, [batchResult, jobStates]);

  const handleDrop = useCallback(e => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter(f =>
      f.name.toLowerCase().endsWith('.tif') || f.name.toLowerCase().endsWith('.tiff')
    );
    setFiles(prev => {
      const next = [...prev, ...dropped].slice(0, 20);
      return next;
    });
  }, []);

  const handleFileChange = e => {
    const selected = Array.from(e.target.files || []).slice(0, 20 - files.length);
    setFiles(prev => [...prev, ...selected].slice(0, 20));
  };

  const removeFile = idx => setFiles(prev => prev.filter((_, i) => i !== idx));

  const handleSubmit = async () => {
    if (!files.length) return;
    setError(null);
    setIsSubmitting(true);
    setJobStates({});
    setBatchResult(null);
    try {
      const result = await startBatchProcessing(files, selectedApp);
      setBatchResult(result);
      // Initialise states as queued
      const initial = {};
      result.jobs.forEach(({ job_id }) => {
        initial[job_id] = { status: 'queued', stage: 'queued', progress: 0 };
      });
      setJobStates(initial);
    } catch (err) {
      setError(err.message || 'Batch submission failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    clearInterval(pollingRef.current);
    setFiles([]);
    setBatchResult(null);
    setJobStates({});
    setError(null);
  };

  const completedCount = Object.values(jobStates).filter(j => j.status === 'completed').length;
  const failedCount    = Object.values(jobStates).filter(j => j.status === 'failed').length;
  const totalJobs      = batchResult?.jobs?.length || 0;
  const isRunning      = batchResult && (completedCount + failedCount) < totalJobs;

  const hw = computeProfile?.hardware;
  const plan = computeProfile?.execution_plan;

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
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <FileStack className="w-5 h-5" />
              </div>
              <h1 className="text-xl md:text-2xl font-black text-white tracking-tight">
                Batch Processing
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                Multi-File Pipeline
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Process up to 20 Sentinel-2 GeoTIFF files simultaneously through PixelSight LDSR-S2
            </p>
          </div>
        </div>

        {/* Hardware Badge */}
        {hw && (
          <button
            onClick={() => setShowProfile(!showProfile)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:border-slate-700 transition-colors"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono">{hw.primary_gpu ? hw.primary_gpu.name : 'CPU only'}</span>
            {showProfile ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        )}
      </header>

      {/* Hardware Profile Panel */}
      {showProfile && computeProfile && (
        <div className="mb-6 p-5 rounded-2xl bg-slate-900 border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-5 text-xs animate-fade-in">
          <div>
            <div className="text-slate-400 font-bold uppercase tracking-wider mb-2">Hardware</div>
            <div className="space-y-1 text-slate-300">
              <div>Device: <span className="font-mono text-cyan-300">{hw.device_type.toUpperCase()}</span></div>
              {hw.primary_gpu && <div>GPU: <span className="font-mono">{hw.primary_gpu.name}</span></div>}
              {hw.primary_gpu && <div>VRAM: <span className="font-mono text-emerald-300">{hw.primary_gpu.vram_gb} GB</span></div>}
              <div>CPU: <span className="font-mono">{hw.cpu_logical_cores} logical cores</span></div>
              {hw.system_ram_gb > 0 && <div>RAM: <span className="font-mono">{hw.system_ram_gb.toFixed(1)} GB</span></div>}
            </div>
          </div>
          <div>
            <div className="text-slate-400 font-bold uppercase tracking-wider mb-2">Execution Plan</div>
            <div className="space-y-1 text-slate-300">
              <div>Strategy: <span className="font-mono text-amber-300">{plan.strategy}</span></div>
              <div>Tile Batch: <span className="font-mono">{plan.tile_batch_size}</span></div>
              <div>Tile Workers: <span className="font-mono">{plan.tile_workers}</span></div>
              <div>FP16: <span className={`font-mono ${plan.fp16_inference ? 'text-emerald-400' : 'text-slate-500'}`}>{plan.fp16_inference ? 'Yes' : 'No'}</span></div>
              <div>Max Active Jobs: <span className="font-mono">{plan.max_active_jobs}</span></div>
            </div>
          </div>
          <div>
            <div className="text-slate-400 font-bold uppercase tracking-wider mb-2">Planner Notes</div>
            <div className="space-y-1">
              {(plan.notes || []).map((n, i) => (
                <div key={i} className="text-slate-400 text-[10px] leading-relaxed">{n}</div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 flex items-start gap-3 text-xs">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5 text-red-400" />
          <div>{error}</div>
          <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-200">✕</button>
        </div>
      )}

      {!batchResult ? (
        /* ── Upload Phase ── */
        <div className="space-y-6">
          {/* Application Selection */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
            <div className="text-sm font-bold text-slate-300 mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              Select Analysis Pipeline
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {APP_OPTIONS.map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setSelectedApp(opt.id)}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    selectedApp === opt.id
                      ? 'border-cyan-500/50 bg-cyan-500/[0.08] shadow-lg shadow-cyan-500/10'
                      : 'border-slate-700 hover:border-slate-600 bg-slate-950/40'
                  }`}
                >
                  <div className="w-2 h-2 rounded-full mb-2" style={{ background: opt.color }} />
                  <div className="text-xs font-bold text-white">{opt.label}</div>
                  <div className="text-[10px] text-slate-400 mt-1 leading-relaxed">{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Drop Zone */}
          <div
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-10 flex flex-col items-center justify-center text-center cursor-pointer ${
              isDragging
                ? 'border-cyan-400 bg-cyan-500/[0.08] shadow-lg shadow-cyan-500/10'
                : 'border-slate-700 hover:border-slate-600 bg-slate-950/30'
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className={`p-4 rounded-2xl mb-4 transition-all ${isDragging ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'}`}>
              <Upload className="w-8 h-8" />
            </div>
            <div className="text-sm font-semibold text-slate-200 mb-1">
              {isDragging ? 'Drop your GeoTIFF files here' : 'Drag & drop Sentinel-2 GeoTIFF files'}
            </div>
            <div className="text-xs text-slate-500">
              Supports .tif / .tiff · Max 20 files · B02, B03, B04, B08 bands required
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".tif,.tiff"
              className="hidden"
              onChange={handleFileChange}
            />
            {files.length > 0 && (
              <div className="mt-4 px-4 py-2 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold">
                {files.length} file{files.length !== 1 ? 's' : ''} selected
              </div>
            )}
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between mb-3">
                <div className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Queued Files ({files.length}/20)
                </div>
                <button
                  onClick={() => setFiles([])}
                  className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                >
                  Clear all
                </button>
              </div>
              <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center justify-between py-2 px-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs">
                    <span className="text-slate-300 truncate flex-1 font-mono">{f.name}</span>
                    <span className="text-slate-500 mx-3 flex-shrink-0">{(f.size / (1024 * 1024)).toFixed(1)} MB</span>
                    <button
                      onClick={() => removeFile(i)}
                      className="text-slate-600 hover:text-red-400 transition-colors flex-shrink-0"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>

              {/* Launch */}
              <div className="pt-3 flex justify-end">
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting || !files.length}
                  className={`flex items-center gap-2.5 px-6 py-3 rounded-xl font-bold text-xs uppercase tracking-wider transition-all shadow-xl ${
                    isSubmitting || !files.length
                      ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                      : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white border border-cyan-500/30 active:scale-95 shadow-cyan-500/20'
                  }`}
                >
                  {isSubmitting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Submitting Batch...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      Start Batch ({files.length} files · {selectedApp})
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* ── Results Phase ── */
        <div className="space-y-6">
          {/* Batch Summary Header */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white">Batch Running</span>
                <span className="font-mono text-[10px] text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
                  {batchResult.batch_id}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span>{completedCount} completed</span>
                <span>·</span>
                <span>{failedCount} failed</span>
                <span>·</span>
                <span>{totalJobs - completedCount - failedCount} pending</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Overall progress */}
              <div className="text-right">
                <div className="text-2xl font-bold font-mono text-cyan-300">
                  {totalJobs > 0 ? Math.round((completedCount / totalJobs) * 100) : 0}%
                </div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Overall</div>
              </div>

              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 transition-colors border border-slate-800"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                New Batch
              </button>
            </div>
          </div>

          {/* Overall Progress Bar */}
          <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${totalJobs > 0 ? (completedCount / totalJobs) * 100 : 0}%`,
                background: 'linear-gradient(90deg, #10b981, #06b6d4)',
                boxShadow: '0 0 12px rgba(6,182,212,0.5)',
              }}
            />
          </div>

          {/* Individual Job Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {batchResult.jobs.map(({ job_id, filename }) => (
              <JobCard
                key={job_id}
                job={jobStates[job_id]}
                jobId={job_id}
                filename={filename}
              />
            ))}
          </div>

          {/* Stats summary when all done */}
          {!isRunning && totalJobs > 0 && (
            <div className="p-5 rounded-2xl bg-emerald-500/[0.06] border border-emerald-500/20">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-bold text-emerald-300">Batch Complete</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-center text-xs">
                <div>
                  <div className="text-2xl font-bold font-mono text-emerald-300">{completedCount}</div>
                  <div className="text-slate-400 mt-0.5">Succeeded</div>
                </div>
                <div>
                  <div className="text-2xl font-bold font-mono text-red-400">{failedCount}</div>
                  <div className="text-slate-400 mt-0.5">Failed</div>
                </div>
                <div>
                  <div className="text-2xl font-bold font-mono text-cyan-300">{totalJobs}</div>
                  <div className="text-slate-400 mt-0.5">Total</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
