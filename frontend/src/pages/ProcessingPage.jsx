import React from 'react';
import { CheckCircle2, Circle, Loader2, Zap, AlertTriangle } from 'lucide-react';

const APP_STAGES = {
  research: [
    { id: 'queued', label: 'Queued', detail: 'Waiting for worker thread', icon: '⏳', progress: 0 },
    { id: 'downloading_data', label: 'Copernicus Download', detail: 'Acquiring B02/B03/B04/B08 from Copernicus Processing API', icon: '📥', progress: 15 },
    { id: 'inspecting', label: 'Inspecting Raster', detail: 'Validating raster metadata, CRS, and band count', icon: '🔍', progress: 20 },
    { id: 'preprocessing', label: 'Preprocessing', detail: 'Normalizing to float32 reflectance, band alignment', icon: '⚙️', progress: 25 },
    { id: 'super_resolution', label: 'LDSR-S2 Super-Resolution', detail: 'Running diffusion model — 100 sampling steps, 4× scale', icon: '🛰️', progress: 40 },
    { id: 'uncertainty_analysis', label: 'Uncertainty Analysis', detail: 'Computing stochastic diffusion variance map', icon: '📊', progress: 75 },
    { id: 'urban_analysis', label: 'Downstream Land-Cover', detail: 'WorldCover-style segmentation on SR output', icon: '🏙️', progress: 85 },
    { id: 'reporting', label: 'Scientific Reporting', detail: 'Writing JSON / Markdown / HTML reports and manifest', icon: '📋', progress: 95 },
    { id: 'completed', label: 'Completed', detail: 'All artifacts ready for download', icon: '✅', progress: 100 },
  ],
  crop: [
    { id: 'queued', label: 'Queued', detail: 'Waiting for worker thread', icon: '⏳', progress: 0 },
    { id: 'downloading_data', label: 'Copernicus Download', detail: 'Acquiring selected Sentinel-2 L2A AOI', icon: '📥', progress: 15 },
    { id: 'inspecting', label: 'Inspecting Raster', detail: 'Validating Sentinel-2 B04 & B08 bands', icon: '🔍', progress: 20 },
    { id: 'preprocessing', label: 'Preprocessing', detail: 'Float32 normalization & profile alignment', icon: '⚙️', progress: 25 },
    { id: 'super_resolution', label: 'LDSR-S2 Super-Resolution', detail: 'Running diffusion model (100 steps, ~2.5m equivalent)', icon: '🛰️', progress: 45 },
    { id: 'uncertainty_analysis', label: 'Uncertainty Analysis', detail: 'Stochastic diffusion variance propagation', icon: '📊', progress: 75 },
    { id: 'crop_analysis', label: 'Vegetation & NDVI Analysis', detail: 'Native vs SR NDVI, MAE/RMSE, stress fractions', icon: '🌱', progress: 88 },
    { id: 'reporting', label: 'Agricultural Reporting', detail: 'Writing JSON / Markdown / HTML reports and manifest', icon: '📋', progress: 95 },
    { id: 'completed', label: 'Completed', detail: 'Crop health maps and metrics ready', icon: '✅', progress: 100 },
  ],
  urban: [
    { id: 'queued', label: 'Queued', detail: 'Waiting for worker thread', icon: '⏳', progress: 0 },
    { id: 'downloading_data', label: 'Copernicus Download', detail: 'Acquiring selected Sentinel-2 L2A AOI', icon: '📥', progress: 15 },
    { id: 'inspecting', label: 'Inspecting Raster', detail: 'Validating 4-band spectral grid', icon: '🔍', progress: 20 },
    { id: 'preprocessing', label: 'Preprocessing', detail: 'Reflectance scaling and georeference preservation', icon: '⚙️', progress: 25 },
    { id: 'super_resolution', label: 'LDSR-S2 Super-Resolution', detail: '4x spatial reconstruction at 100 diffusion steps', icon: '🛰️', progress: 45 },
    { id: 'uncertainty_analysis', label: 'Uncertainty Analysis', detail: 'Spatial confidence estimation', icon: '📊', progress: 75 },
    { id: 'urban_analysis', label: 'Urban Land-Cover Segmentation', detail: 'UNet WorldCover inference & built-up mask generation', icon: '🏙️', progress: 88 },
    { id: 'reporting', label: 'Urban Reporting', detail: 'Writing JSON / Markdown / HTML reports and manifest', icon: '📋', progress: 95 },
    { id: 'completed', label: 'Completed', detail: 'Building cluster and land-cover maps ready', icon: '✅', progress: 100 },
  ],
  disaster: [
    { id: 'queued', label: 'Queued', detail: 'Waiting for worker thread', icon: '⏳', progress: 0 },
    { id: 'pre_event_acquisition', label: 'Pre-Event Acquisition', detail: 'Downloading pre-event Sentinel-2 AOI from Copernicus', icon: '📥', progress: 10 },
    { id: 'post_event_acquisition', label: 'Post-Event Acquisition', detail: 'Downloading post-event Sentinel-2 AOI from Copernicus', icon: '📥', progress: 25 },
    { id: 'preprocessing', label: 'Preprocessing', detail: 'Normalizing both temporal scenes', icon: '⚙️', progress: 35 },
    { id: 'alignment', label: 'Geospatial Alignment', detail: 'Harmonizing CRS, pixel grid, and bounding extents', icon: '📐', progress: 45 },
    { id: 'super_resolution_pre', label: 'Pre-Event Super-Resolution', detail: 'Running LDSR-S2 on pre-event baseline (100 steps)', icon: '🛰️', progress: 55 },
    { id: 'super_resolution_post', label: 'Post-Event Super-Resolution', detail: 'Running LDSR-S2 on post-event scene (100 steps)', icon: '🛰️', progress: 70 },
    { id: 'uncertainty', label: 'Uncertainty Estimation', detail: 'Computing dual-acquisition diffusion variance', icon: '📊', progress: 80 },
    { id: 'change_analysis', label: 'Spectral Change Detection', detail: 'Evaluating spectral change vector & uncertainty reliability', icon: '🔥', progress: 90 },
    { id: 'reporting', label: 'Disaster Assessment Reporting', detail: 'Writing JSON / Markdown / HTML reports and manifest', icon: '📋', progress: 95 },
    { id: 'completed', label: 'Completed', detail: 'Temporal change maps and impact masks ready', icon: '✅', progress: 100 },
  ],
};

function StageRow({ stage, status }) {
  return (
    <div className="flex gap-4 items-start py-3 border-b border-white/[0.04] last:border-0">
      <div
        className={`
          w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm mt-0.5
          ${status === 'done' ? 'bg-emerald-500/15 border border-emerald-500/30' : ''}
          ${status === 'active' ? 'bg-cyan-500/15 border border-cyan-500/40' : ''}
          ${status === 'wait' ? 'bg-white/[0.03] border border-white/[0.08]' : ''}
          ${status === 'error' ? 'bg-red-500/15 border border-red-500/30' : ''}
        `}
      >
        {status === 'done' && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
        {status === 'active' && <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />}
        {status === 'wait' && <Circle className="w-4 h-4 text-slate-700" />}
        {status === 'error' && <AlertTriangle className="w-4 h-4 text-red-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <div
          className={`text-sm font-semibold
            ${status === 'done' ? 'text-emerald-400' : ''}
            ${status === 'active' ? 'text-white' : ''}
            ${status === 'wait' ? 'text-slate-600' : ''}
            ${status === 'error' ? 'text-red-400' : ''}
          `}
        >
          <span className="mr-2">{stage.icon}</span>
          {stage.label}
        </div>
        <div className="text-xs text-slate-500 mt-0.5">{stage.detail}</div>
      </div>
      {status === 'active' && (
        <div className="text-xs font-mono text-cyan-400 tabular-nums mt-1">{stage.progress}%</div>
      )}
      {status === 'done' && (
        <div className="text-xs font-mono text-emerald-500/70 mt-1">done</div>
      )}
    </div>
  );
}

export default function ProcessingPage({ currentStep, job }) {
  const application = job?.application || 'research';
  const stages = APP_STAGES[application] || APP_STAGES.research;

  const stepLower = (currentStep || job?.stage || 'queued').toLowerCase().replace(/-/g, '_');
  const stageIndex = Math.max(0, stages.findIndex(s => s.id === stepLower));
  const progress = job?.progress ?? (stepLower === 'queued' ? 0 : 5);
  const isFailed = job?.status === 'failed';

  const currentStage = stages[stageIndex] || stages[0];
  const estimatedPhase = currentStage?.label ?? 'Processing…';

  return (
    <main
      className="min-h-screen flex items-center justify-center px-6"
      style={{
        background:
          'radial-gradient(ellipse 65% 50% at 50% 45%, rgba(6,182,212,0.07), transparent 70%), #050a14',
      }}
    >
      <section className="w-full max-w-2xl anim-fade-up">
        {/* Header */}
        <div className="text-center mb-10">
          <div
            className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-6"
            style={{
              background: 'rgba(6,182,212,0.12)',
              border: '1px solid rgba(6,182,212,0.3)',
            }}
          >
            <Zap className="w-8 h-8 text-cyan-400" />
          </div>
          <h1 className="text-3xl font-bold text-white">Running the Real Pipeline</h1>
          <p className="text-slate-400 mt-2">
            {job?.job_id ? `Job ${job.job_id} · ${application.toUpperCase()}` : 'Submitting job…'}
          </p>
          <div className="mt-2 text-sm text-cyan-300 font-mono">{estimatedPhase}</div>
        </div>

        {/* Progress Bar */}
        <div className="card p-6 mb-6">
          <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
            <span>Overall Progress</span>
            <span className="font-mono text-cyan-400 font-bold">{Math.round(progress)}%</span>
          </div>
          <div className="h-2.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isFailed ? 'bg-red-500' : 'bg-gradient-to-r from-cyan-500 to-blue-500'
              }`}
              style={{ width: `${Math.max(3, Math.min(100, progress))}%` }}
            />
          </div>

          {/* Stage list */}
          <div className="mt-6 divide-y divide-white/[0.04]">
            {stages.map((stage, idx) => {
              let status = 'wait';
              if (isFailed && idx === stageIndex) status = 'error';
              else if (idx < stageIndex) status = 'done';
              else if (idx === stageIndex) status = 'active';

              return <StageRow key={stage.id} stage={stage} status={status} />;
            })}
          </div>

          {isFailed && (
            <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300 space-y-1">
              <div className="font-bold flex items-center gap-1.5 text-red-400">
                <AlertTriangle className="w-4 h-4" /> Pipeline Error
              </div>
              <div className="font-mono break-all">{job?.error || 'Processing encountered an unhandled exception.'}</div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
