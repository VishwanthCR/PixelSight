import React, { useRef, useState } from 'react';
import { CheckCircle2, FileImage, ScanSearch, Upload, Zap } from 'lucide-react';

const fields = [
  ['Bands', inspection => inspection?.band_names?.join(' / ') || '—'],
  ['Dimensions', inspection => inspection ? `${inspection.width} × ${inspection.height}` : '—'],
  ['Resolution', inspection => inspection?.resolution ? `${inspection.resolution[0]}m × ${inspection.resolution[1]}m` : '—'],
  ['CRS', inspection => inspection?.crs || '—'],
  ['Datatype', inspection => inspection?.dtype || '—'],
];

export default function LandingPage({ file, inspection, onSelectFile, onProcess, health }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function choose(nextFile) {
    if (nextFile) onSelectFile(nextFile);
  }

  return (
    <main className="min-h-screen px-5 py-6 md:px-10" style={{ background: 'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(6,182,212,0.11), transparent 70%), #050a14' }}>
      <header className="max-w-6xl mx-auto flex items-center justify-between mb-16">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#0891b2,#1d4ed8)' }}><Zap className="w-5 h-5 text-white" /></div>
          <div><div className="font-bold text-white tracking-tight">PixelSight</div><div className="text-[10px] text-cyan-400 uppercase tracking-[0.22em]">LDSR-S2 framework</div></div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-400' : health?.status === 'offline' ? 'bg-red-400' : 'bg-amber-400'}`} />
          <span className="text-slate-500">
            {health?.status === 'ok' ? 'Backend online' : health?.status === 'offline' ? 'Backend offline' : 'Checking backend'}
          </span>
        </div>
      </header>

      <section className="max-w-6xl mx-auto grid lg:grid-cols-[1.05fr_0.95fr] gap-10 items-start">
        <div className="pt-4 anim-fade-up">
          <div className="text-xs text-cyan-400 font-bold uppercase tracking-[0.24em] mb-5">Satellite image processing</div>
          <h1 className="text-4xl md:text-6xl font-bold text-white leading-[1.05] tracking-tight mb-6">From native pixels<br /><span className="text-cyan-300">to sharper evidence.</span></h1>
          <p className="text-slate-400 text-lg leading-relaxed max-w-xl">Upload a PNG, JPEG, or compatible Sentinel-2 GeoTIFF. PixelSight converts ordinary images into the model's four-band input, runs the real LDSR-S2 model, and returns a georeferenced 4x super-resolved representation (~2.5m equivalent).</p>
          <div className="mt-8 flex flex-wrap gap-3 text-xs text-slate-500"><span className="tag">PNG · JPEG · GeoTIFF</span><span className="tag">4-band model adapter</span><span className="tag">urban planning counts</span></div>
        </div>

        <div className="space-y-5 anim-fade-up" style={{ animationDelay: '0.08s' }}>
          <div className={`upload-zone p-8 md:p-12 text-center ${dragging ? 'drag-over' : ''}`} onClick={() => inputRef.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]); }}>
            <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" className="hidden" onChange={event => { choose(event.target.files?.[0]); event.target.value = ''; }} />
            <div className="w-14 h-14 mx-auto rounded-2xl flex items-center justify-center mb-5" style={{ background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.3)' }}><Upload className="w-7 h-7 text-cyan-400" /></div>
            <div className="text-lg font-semibold text-white">{file ? file.name : 'Drop a GeoTIFF scene here'}</div>
            <div className="text-sm text-slate-500 mt-2">{file ? 'Select another file to replace it' : 'or click to browse · PNG, JPEG, or GeoTIFF'}</div>
          </div>

          {inspection && (
            <div className="card p-5 anim-fade-in">
              <div className="flex items-center justify-between mb-4"><div className="flex items-center gap-2 text-sm font-semibold text-white"><ScanSearch className="w-4 h-4 text-cyan-400" /> Inspection</div><span className={`text-xs font-semibold ${inspection.compatible ? 'text-emerald-400' : 'text-red-400'}`}>{inspection.compatible ? 'Compatible' : 'Rejected'}</span></div>
              <div className="grid grid-cols-2 gap-3">{fields.map(([label, value]) => <div key={label} className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.035)' }}><div className="text-[10px] uppercase tracking-wider text-slate-600">{label}</div><div className="text-xs text-slate-300 font-mono mt-1 break-all">{value(inspection)}</div></div>)}</div>
              {inspection.errors?.length > 0 && <div className="mt-4 text-xs text-red-300 space-y-1">{inspection.errors.map(error => <div key={error}>{error}</div>)}</div>}
              {inspection.warnings?.length > 0 && <div className="mt-4 text-xs text-amber-300 space-y-1">{inspection.warnings.map(warning => <div key={warning}>{warning}</div>)}</div>}
            </div>
          )}

          <button className="btn-primary w-full py-4" disabled={!inspection?.compatible} onClick={onProcess}><CheckCircle2 className="w-5 h-5" /> Run real LDSR-S2 processing</button>
          <div className="text-center text-xs text-slate-600">The backend performs inference. The browser only displays returned artifacts.</div>
        </div>
      </section>
    </main>
  );
}
