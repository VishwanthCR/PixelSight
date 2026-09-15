import React from 'react';
import { FileText, Copy, Check, GitCommit, ShieldCheck } from 'lucide-react';

export default function ImpactReport({ data }) {
  const [copied, setCopied] = React.useState(false);

  if (!data) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(data.impact_report || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lineage = data.lineage || {};

  return (
    <div className="glass-card rounded-2xl p-6 mb-8">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-400" />
            Executive Impact Analysis Report & Lineage Traceability
          </h3>
          <p className="text-xs text-slate-400">
            Synthesized narrative report tied to the selected use-case lens and model version metadata
          </p>
        </div>

        <button
          onClick={handleCopy}
          className="px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-all flex items-center gap-1.5"
        >
          {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4 text-slate-400" />}
          {copied ? 'Copied Report' : 'Copy Text'}
        </button>
      </div>

      {/* Model Lineage Traceability Badge Card */}
      <div className="bg-slate-900/80 border border-blue-500/30 rounded-xl p-4 mb-6 text-xs font-mono space-y-2">
        <div className="flex items-center gap-2 text-blue-300 font-bold font-sans text-sm border-b border-slate-800 pb-2">
          <GitCommit className="w-4 h-4 text-blue-400" />
          Model & Dataset Traceability Metadata (Judge Q&A Audit)
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-slate-300 pt-1">
          <div>
            <span className="text-slate-500 block">Model Version:</span>
            <span className="text-white font-semibold">{lineage.model_version || 'Teammate Checkpoint'}</span>
          </div>
          <div>
            <span className="text-slate-500 block">Checkpoint Status:</span>
            <span className={lineage.checkpoint_status?.includes('Loaded') ? 'text-emerald-400' : 'text-amber-400'}>
              {lineage.checkpoint_status || 'Pending Checkpoint'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block">Source Region:</span>
            <span className="text-cyan-300">{lineage.source_region || 'custom'}</span>
          </div>
          <div>
            <span className="text-slate-500 block">Patch Index / Input:</span>
            <span className="text-purple-300">{lineage.patch_index ?? 'Upload'} ({lineage.input_shape || '4 ch'})</span>
          </div>
        </div>
      </div>

      {/* Narrative Report Content */}
      <div className="prose prose-invert max-w-none text-slate-300 text-sm leading-relaxed whitespace-pre-wrap bg-slate-950/60 p-5 rounded-xl border border-slate-800">
        {data.impact_report}
      </div>

    </div>
  );
}
