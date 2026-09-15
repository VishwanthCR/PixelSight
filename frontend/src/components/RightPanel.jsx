/* PixelSight — Right Panel: Metrics / Analysis / Uncertainty / Layers */

import React, { useState } from 'react';
import {
  BarChart2, ShieldAlert, Layers, Download,
  ChevronRight, AlertCircle, CheckCircle2, Clock,
  TrendingUp, Activity, Cpu, GitCommit, FileText
} from 'lucide-react';

const TABS = [
  { id: 'metrics',     label: 'Metrics',      Icon: BarChart2   },
  { id: 'uncertainty', label: 'Uncertainty',  Icon: ShieldAlert },
  { id: 'spectral',    label: 'Spectral',     Icon: Activity    },
  { id: 'report',      label: 'Report',       Icon: FileText    },
];

function Toggle({ on, onChange }) {
  return (
    <div
      className={`toggle-track ${on ? 'on' : ''}`}
      onClick={() => onChange(!on)}
      style={{ background: on ? '#06b6d4' : '#1e293b' }}
    >
      <div className="toggle-thumb" />
    </div>
  );
}

function MetricCard({ label, value, unit, icon: Icon, color = '#06b6d4', sub }) {
  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
          <div className="text-xl font-bold font-mono" style={{ color }}>
            {value ?? '—'}{unit && <span className="text-sm ml-1 font-normal text-slate-400">{unit}</span>}
          </div>
          {sub && <div className="text-[10px] text-slate-600 mt-0.5">{sub}</div>}
        </div>
        {Icon && <Icon className="w-4 h-4 mt-0.5 opacity-40" style={{ color }} />}
      </div>
    </div>
  );
}

export default function RightPanel({ data }) {
  const [activeTab, setActiveTab] = useState('metrics');
  const [layers, setLayers] = useState({
    before: true, after: true, uncertainty: false, residual: false,
  });

  const m = data?.metrics || {};
  const s = data?.spectral || {};
  const lin = data?.lineage || {};
  const isIoUPending = !m.iou || m.iou === 'pending';

  return (
    <aside className="flex flex-col h-full border-l border-white/5 overflow-hidden" style={{ width: 296, minWidth: 296 }}>

      {/* ── Tab bar ── */}
      <div className="flex border-b border-white/5 flex-shrink-0">
        {TABS.map(t => (
          <button key={t.id} className={`tab-btn ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">

        {/* ════ METRICS TAB ════ */}
        {activeTab === 'metrics' && (
          <>
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Quality Metrics</div>
              {!data ? (
                <p className="text-xs text-slate-600 italic">Run analysis to see metrics</p>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  <MetricCard label="PSNR" value={m.psnr} unit="dB" icon={TrendingUp} />
                  <MetricCard label="SSIM" value={m.ssim} icon={Activity} color="#a78bfa" />
                  <MetricCard label="Resolution Gain" value={m.resolution_multiplier} icon={Cpu} color="#10b981" />
                  <MetricCard
                    label="IoU (Mask)"
                    value={isIoUPending ? null : m.iou}
                    icon={isIoUPending ? Clock : CheckCircle2}
                    color={isIoUPending ? '#f59e0b' : '#10b981'}
                    sub={isIoUPending ? 'Pending GT mask' : 'vs reference mask'}
                  />
                </div>
              )}
            </div>

            {/* Lineage / Traceability */}
            {data && (
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Model Lineage</div>
                <div className="space-y-2 text-xs font-mono">
                  {[
                    { label: 'Checkpoint', value: lin.checkpoint_status, warn: lin.checkpoint_status?.includes('Awaiting') },
                    { label: 'Region',     value: lin.source_region },
                    { label: 'Scale',      value: lin.spatial_scale },
                    { label: 'Bands',      value: lin.input_shape },
                  ].map(row => (
                    <div key={row.label} className="flex items-start gap-2 py-1.5 border-b border-white/[0.04]">
                      <span className="text-slate-600 w-20 flex-shrink-0">{row.label}</span>
                      <span className={`flex-1 truncate ${row.warn ? 'text-amber-400' : 'text-slate-300'}`}>{row.value || '—'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Layer Toggles */}
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">Layer Visibility</div>
              <div className="space-y-3">
                {[
                  { id: 'before',      label: 'Before (Original LR)',  sub: 'Source satellite imagery' },
                  { id: 'after',       label: 'After (SR Enhanced)',    sub: 'Neural super-resolution output' },
                  { id: 'uncertainty', label: 'Uncertainty Map',        sub: 'MC Dropout std-dev heatmap' },
                  { id: 'residual',    label: 'Residual Detail Map',    sub: 'SR − Bicubic difference' },
                ].map(l => (
                  <div key={l.id} className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-slate-300 font-medium">{l.label}</div>
                      <div className="text-[10px] text-slate-600">{l.sub}</div>
                    </div>
                    <Toggle on={layers[l.id]} onChange={v => setLayers(prev => ({ ...prev, [l.id]: v }))} />
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* ════ UNCERTAINTY TAB ════ */}
        {activeTab === 'uncertainty' && (
          <>
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">MC Dropout Uncertainty</div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Per-pixel standard deviation across {data?.lineage ? 'N' : '—'} stochastic forward passes. Bright areas = high model uncertainty.
              </p>
            </div>

            {!data ? (
              <p className="text-xs text-slate-600 italic">Run analysis first</p>
            ) : (
              <>
                <div className="rounded-xl overflow-hidden border border-white/[0.06]">
                  <img src={data.uncertainty_map_b64} alt="Uncertainty Map" className="w-full object-contain" />
                  {/* Colorbar */}
                  <div className="flex items-center justify-between px-3 py-2 bg-slate-900/80 text-[10px] font-mono">
                    <span className="text-emerald-400">0.0 Low</span>
                    <div className="flex-1 mx-3 h-1.5 rounded bg-gradient-to-r from-black via-purple-600 to-yellow-300" />
                    <span className="text-amber-400">High σ</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <MetricCard label="Mean σ" value={(m.mean_uncertainty ?? 0).toFixed(4)} color="#f59e0b" icon={Activity} />
                  <MetricCard label="Peak σ" value={(m.max_uncertainty ?? 0).toFixed(4)} color="#ef4444" icon={ShieldAlert} />
                  <div className="col-span-2 metric-card">
                    <div className="text-[10px] text-slate-500 mb-1">Model Confidence</div>
                    <div className="text-xl font-bold font-mono text-emerald-400">
                      {Math.max(0, Math.min(100, (1 - (m.mean_uncertainty ?? 0) * 3) * 100)).toFixed(1)}%
                    </div>
                    <div className="mt-2 h-1.5 rounded-full bg-slate-800">
                      <div className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all"
                        style={{ width: `${Math.max(0, Math.min(100, (1 - (m.mean_uncertainty ?? 0) * 3) * 100)).toFixed(1)}%` }} />
                    </div>
                  </div>
                </div>
              </>
            )}
          </>
        )}

        {/* ════ SPECTRAL TAB ════ */}
        {activeTab === 'spectral' && (
          <>
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Spectral Consistency</div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                NDVI (vegetation) and NDWI (water) index comparison verifies spectral fidelity after SR enhancement.
              </p>
            </div>

            {!data ? (
              <p className="text-xs text-slate-600 italic">Run analysis first</p>
            ) : (
              <>
                {/* NDVI */}
                <div>
                  <div className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider mb-2">NDVI — Vegetation Index</div>
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div>
                      <div className="text-[9px] text-slate-500 mb-1">LR (Bicubic)</div>
                      <img src={data.ndvi_lr_b64} alt="NDVI LR" className="w-full rounded-lg border border-white/[0.06]" />
                    </div>
                    <div>
                      <div className="text-[9px] text-cyan-400 mb-1">SR (Enhanced)</div>
                      <img src={data.ndvi_sr_b64} alt="NDVI SR" className="w-full rounded-lg border border-cyan-500/20" />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-mono">
                    <div className="metric-card py-2"><div className="text-slate-500">LR Mean</div><div className="text-white font-bold">{s.mean_ndvi_lr?.toFixed(3) ?? '—'}</div></div>
                    <div className="metric-card py-2"><div className="text-cyan-400">SR Mean</div><div className="text-cyan-300 font-bold">{s.mean_ndvi_sr?.toFixed(3) ?? '—'}</div></div>
                    <div className="metric-card py-2"><div className="text-slate-500">Δ</div><div className={`font-bold ${(s.ndvi_delta ?? 0) >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>{s.ndvi_delta != null ? `${s.ndvi_delta >= 0 ? '+' : ''}${s.ndvi_delta.toFixed(3)}` : '—'}</div></div>
                  </div>
                </div>

                {/* NDWI */}
                <div>
                  <div className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider mb-2">NDWI — Water Index</div>
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div>
                      <div className="text-[9px] text-slate-500 mb-1">LR (Bicubic)</div>
                      <img src={data.ndwi_lr_b64} alt="NDWI LR" className="w-full rounded-lg border border-white/[0.06]" />
                    </div>
                    <div>
                      <div className="text-[9px] text-cyan-400 mb-1">SR (Enhanced)</div>
                      <img src={data.ndwi_sr_b64} alt="NDWI SR" className="w-full rounded-lg border border-cyan-500/20" />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-mono">
                    <div className="metric-card py-2"><div className="text-slate-500">LR Mean</div><div className="text-white font-bold">{s.mean_ndwi_lr?.toFixed(3) ?? '—'}</div></div>
                    <div className="metric-card py-2"><div className="text-cyan-400">SR Mean</div><div className="text-cyan-300 font-bold">{s.mean_ndwi_sr?.toFixed(3) ?? '—'}</div></div>
                    <div className="metric-card py-2"><div className="text-slate-500">Δ</div><div className={`font-bold ${(s.ndwi_delta ?? 0) >= 0 ? 'text-cyan-400' : 'text-amber-400'}`}>{s.ndwi_delta != null ? `${s.ndwi_delta >= 0 ? '+' : ''}${s.ndwi_delta.toFixed(3)}` : '—'}</div></div>
                  </div>
                </div>

                {/* Residual map */}
                <div>
                  <div className="text-[10px] font-semibold text-purple-400 uppercase tracking-wider mb-2">Residual Detail Map (SR − Bicubic)</div>
                  <img src={data.residual_map_b64} alt="Residual" className="w-full rounded-lg border border-purple-500/20" />
                </div>
              </>
            )}
          </>
        )}

        {/* ════ REPORT TAB ════ */}
        {activeTab === 'report' && (
          <>
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Executive Impact Report</div>
            </div>

            {!data ? (
              <p className="text-xs text-slate-600 italic">Run analysis to generate report</p>
            ) : (
              <>
                {/* Lineage Badge */}
                <div className="rounded-xl border border-blue-500/20 p-3 bg-blue-500/[0.04] text-xs space-y-1.5 font-mono">
                  <div className="flex items-center gap-1.5 text-blue-300 font-bold font-sans text-[11px] pb-1 border-b border-white/[0.04]">
                    <GitCommit className="w-3.5 h-3.5" />
                    Traceability Audit
                  </div>
                  <div className="flex justify-between"><span className="text-slate-600">Model</span><span className="text-slate-300 truncate max-w-[130px]">{lin.model_version ?? '—'}</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Region</span><span className="text-cyan-300">{lin.source_region ?? '—'}</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Scale</span><span className="text-emerald-300">{lin.spatial_scale ?? '—'}</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Checkpoint</span><span className={lin.checkpoint_status?.includes('Awaiting') ? 'text-amber-400' : 'text-emerald-400'}>{lin.checkpoint_status?.includes('Awaiting') ? 'Scaffolded' : 'Loaded'}</span></div>
                </div>

                {/* Narrative report */}
                <div className="text-[11px] text-slate-300 leading-relaxed bg-slate-900/50 rounded-xl p-3 border border-white/[0.04] whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {data.impact_report}
                </div>

                {/* Export buttons */}
                <div>
                  <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Export</div>
                  <div className="space-y-2">
                    {[
                      { label: 'SR Enhanced PNG', sub: 'Super-resolution output image' },
                      { label: 'Uncertainty Map', sub: 'MC Dropout σ heatmap overlay' },
                      { label: 'Analysis Report', sub: 'Metrics + narrative + lineage' },
                    ].map(e => (
                      <button key={e.label} className="w-full flex items-center gap-3 px-3 py-2 rounded-lg border border-slate-800 hover:border-slate-700 hover:bg-white/[0.02] text-left transition-all">
                        <Download className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                        <div>
                          <div className="text-xs text-slate-300">{e.label}</div>
                          <div className="text-[10px] text-slate-600">{e.sub}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </>
        )}

      </div>
    </aside>
  );
}
