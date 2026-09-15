import React from 'react';
import { Gauge, CheckCircle, Clock } from 'lucide-react';

export default function MetricsPanel({ data }) {
  if (!data) return null;

  const metrics = data.metrics || {};
  const isIoUPending = metrics.iou === 'pending' || metrics.iou_status === 'pending';

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Gauge className="w-5 h-5 text-blue-400" />
            Quantitative Evaluation Metrics
          </h3>
          <p className="text-xs text-slate-400">
            Fidelity metrics vs baseline bicubic interpolation and ground truth mask validation
          </p>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* PSNR Card */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="text-xs text-slate-400 font-semibold mb-1">Peak Signal-to-Noise Ratio (PSNR)</div>
          <div className="text-2xl font-bold font-mono text-blue-400">
            {metrics.psnr ? `${metrics.psnr} dB` : 'N/A'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Reconstruction signal fidelity</p>
        </div>

        {/* SSIM Card */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="text-xs text-slate-400 font-semibold mb-1">Structural Similarity (SSIM)</div>
          <div className="text-2xl font-bold font-mono text-purple-400">
            {metrics.ssim ?? 'N/A'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Luminance & structure preservation</p>
        </div>

        {/* IoU Card (Strictly adhering to Constraint 4) */}
        <div className={`rounded-xl p-4 border ${
          isIoUPending ? 'bg-amber-950/20 border-amber-500/30' : 'bg-slate-900/80 border-slate-800'
        }`}>
          <div className="text-xs text-slate-400 font-semibold mb-1 flex items-center justify-between">
            <span>Mask IoU (Intersection over Union)</span>
            {isIoUPending ? (
              <Clock className="w-3.5 h-3.5 text-amber-400" />
            ) : (
              <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
            )}
          </div>
          <div className="text-xl font-bold font-mono text-amber-300">
            {isIoUPending ? 'Pending GT Mask' : metrics.iou}
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            {isIoUPending ? 'Awaiting reference ground truth segmentation mask' : 'Validated vs reference mask'}
          </p>
        </div>

        {/* Resolution Scale Card */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
          <div className="text-xs text-slate-400 font-semibold mb-1">Spatial Resolution Factor</div>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {metrics.resolution_multiplier || '2x'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">10m GSD → 5m GSD upscaling</p>
        </div>

      </div>

    </div>
  );
}
