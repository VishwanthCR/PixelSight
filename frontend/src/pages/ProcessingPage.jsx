import React from 'react';
import { CheckCircle2, Circle, Loader2, Zap } from 'lucide-react';

const STAGES = [
  ['queued', 'Queued', 'Waiting for the worker'],
  ['inspecting', 'Inspecting', 'Validating raster metadata and bands'],
  ['preprocessing', 'Preprocessing', 'Preparing float32 reflectance input'],
  ['super_resolution', 'Super-resolution', 'Running LDSR-S2 with 100 steps'],
  ['reporting', 'Reporting', 'Writing outputs and report'],
  ['completed', 'Completed', 'Artifacts are ready'],
];

export default function ProcessingPage({ currentStep, job }) {
  const index = Math.max(0, STAGES.findIndex(([id]) => id === currentStep));
  const progress = job?.progress ?? (currentStep === 'queued' ? 0 : 5);
  return <main className="min-h-screen flex items-center justify-center px-6" style={{ background: 'radial-gradient(ellipse 65% 50% at 50% 45%, rgba(6,182,212,0.07), transparent 70%), #050a14' }}>
    <section className="w-full max-w-2xl anim-fade-up">
      <div className="text-center mb-10"><div className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-6" style={{ background: 'rgba(6,182,212,0.12)', border: '1px solid rgba(6,182,212,0.3)' }}><Zap className="w-8 h-8 text-cyan-400" /></div><h1 className="text-3xl font-bold text-white">Running the real pipeline</h1><p className="text-slate-400 mt-2">{job?.job_id ? `Job ${job.job_id}` : 'Submitting job...'}</p></div>
      <div className="flex justify-between text-xs mb-2"><span className="text-slate-500">Backend progress</span><span className="font-mono text-cyan-400">{Math.round(progress)}%</span></div><div className="h-2 rounded-full overflow-hidden mb-10" style={{ background: '#0d1a2d' }}><div className="progress-bar" style={{ width: `${progress}%` }} /></div>
      <div className="space-y-1">{STAGES.map(([id, label, detail], stageIndex) => { const done = currentStep === 'completed' || stageIndex < index; const active = id === currentStep && currentStep !== 'completed'; return <div key={id} className="flex gap-4 items-start py-3"><div className={`step-dot ${done ? 'done' : active ? 'active' : 'wait'}`}>{done ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : active ? <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" /> : <Circle className="w-4 h-4 text-slate-700" />}</div><div><div className={`text-sm font-semibold ${done ? 'text-emerald-400' : active ? 'text-white' : 'text-slate-600'}`}>{label}</div><div className="text-xs text-slate-500 mt-1">{detail}</div></div></div>})}</div>
      {job?.status === 'failed' && <div className="mt-8 p-4 rounded-xl text-sm text-red-300" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}>{job.error}</div>}
    </section>
  </main>;
}
