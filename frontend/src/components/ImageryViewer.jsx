/* PixelSight — Center Imagery Viewer with Before/After Drag Slider */

import React, { useState, useRef, useCallback } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

export default function ImageryViewer({ data, loading }) {
  const [sliderX, setSliderX] = useState(50); // percent
  const [zoom, setZoom] = useState(100);
  const [bandMode, setBandMode] = useState('rgb');
  const isDragging = useRef(false);
  const containerRef = useRef();

  /* ── Drag logic ── */
  const onMouseDown = useCallback(e => {
    e.preventDefault();
    isDragging.current = true;
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const onMouseMove = useCallback(e => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.min(Math.max(((e.clientX - rect.left) / rect.width) * 100, 5), 95);
    setSliderX(x);
  }, []);

  const onMouseUp = useCallback(() => {
    isDragging.current = false;
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }, [onMouseMove]);

  /* Touch */
  const onTouchMove = useCallback(e => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.min(Math.max(((e.touches[0].clientX - rect.left) / rect.width) * 100, 5), 95);
    setSliderX(x);
  }, []);

  const lrSrc = bandMode === 'rgb' ? data?.lr_image_b64 : data?.lr_false_color_b64;
  const srSrc = bandMode === 'rgb' ? data?.sr_image_b64 : data?.sr_false_color_b64;

  return (
    <div className="flex-1 flex flex-col h-full min-w-0">

      {/* ── Toolbar bar ── */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-white/5 flex-shrink-0">
        <div>
          <div className="text-sm font-semibold text-white">Imagery Viewer</div>
          <div className="text-[11px] text-slate-500">
            {data ? 'Drag split handle · scroll to zoom' : 'Upload or select a demo patch to begin'}
          </div>
        </div>

        {/* Band toggle */}
        <div className="flex items-center gap-2">
          <div className="flex bg-slate-900/80 border border-slate-800 rounded-lg p-0.5 text-[10px]">
            <button
              onClick={() => setBandMode('rgb')}
              className={`px-3 py-1.5 rounded-md transition-all font-semibold ${bandMode === 'rgb' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-500 hover:text-slate-300'}`}
            >
              True Color
            </button>
            <button
              onClick={() => setBandMode('false_color')}
              className={`px-3 py-1.5 rounded-md transition-all font-semibold ${bandMode === 'false_color' ? 'bg-purple-500/20 text-purple-300' : 'text-slate-500 hover:text-slate-300'}`}
            >
              False Color (NIR)
            </button>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 border border-slate-800 rounded-lg px-2 py-1.5 bg-slate-900/80 text-[10px] font-mono text-slate-400">
            <button onClick={() => setZoom(z => Math.max(50, z - 25))} className="hover:text-white transition-colors"><ZoomOut className="w-3.5 h-3.5" /></button>
            <span className="w-8 text-center">{zoom}%</span>
            <button onClick={() => setZoom(z => Math.min(200, z + 25))} className="hover:text-white transition-colors"><ZoomIn className="w-3.5 h-3.5" /></button>
            <button onClick={() => setZoom(100)} className="hover:text-white transition-colors ml-1"><RotateCcw className="w-3 h-3" /></button>
          </div>
        </div>
      </div>

      {/* ── Main Viewer Area ── */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center" style={{ background: '#070d18' }}>

        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10 gap-4">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 animate-ping" />
              <div className="absolute inset-0 rounded-full border-2 border-t-cyan-400 border-transparent animate-spin" />
            </div>
            <div className="text-sm text-slate-400">Running SR Pipeline...</div>
          </div>
        )}

        {!data && !loading && (
          <div className="flex flex-col items-center gap-3 opacity-30">
            <div className="w-24 h-24 rounded-2xl border-2 border-dashed border-slate-700 flex items-center justify-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="3"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
            </div>
            <span className="text-slate-600 text-sm">No imagery loaded</span>
          </div>
        )}

        {data && (
          <div
            ref={containerRef}
            className="relative select-none"
            style={{
              width: `${zoom}%`,
              maxWidth: '100%',
              aspectRatio: '1 / 1',
              cursor: 'col-resize',
            }}
            onTouchMove={onTouchMove}
          >
            {/* SR image (right side — "After") */}
            <img
              src={srSrc}
              alt="Super-Resolved"
              className="absolute inset-0 w-full h-full object-cover viewer-image rounded-xl"
              draggable={false}
            />

            {/* LR image (left side — "Before") clipped */}
            <div
              className="absolute inset-0 overflow-hidden rounded-xl"
              style={{ width: `${sliderX}%` }}
            >
              <img
                src={lrSrc}
                alt="Original LR"
                className="absolute top-0 left-0 h-full viewer-image rounded-xl"
                style={{ width: `${10000 / sliderX}%`, maxWidth: 'none' }}
                draggable={false}
              />
            </div>

            {/* Before / After labels */}
            <span className="absolute top-3 left-3 text-[10px] font-bold text-white/80 px-2 py-1 rounded-md"
              style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)' }}>
              ◀ BEFORE (10m LR)
            </span>
            <span className="absolute top-3 right-3 text-[10px] font-bold text-cyan-300/90 px-2 py-1 rounded-md"
              style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)' }}>
              AFTER (SR) ▶
            </span>

            {/* Resolution badge */}
            <div className="absolute bottom-3 left-3 text-[10px] px-2 py-1 rounded-md font-mono"
              style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)', color: '#10b981' }}>
              {data.lineage?.spatial_scale || '2× Enhanced'}
            </div>

            {/* Split handle */}
            <div
              onMouseDown={onMouseDown}
              className="absolute top-0 bottom-0 flex items-center justify-center compare-slider z-20"
              style={{ left: `${sliderX}%`, transform: 'translateX(-50%)', width: 32 }}
            >
              <div className="w-px h-full bg-white/50 absolute" />
              <div
                className="relative z-10 w-8 h-8 rounded-full flex items-center justify-center"
                style={{ background: 'rgba(15,30,55,0.9)', border: '2px solid rgba(255,255,255,0.3)', backdropFilter: 'blur(8px)' }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <polyline points="9 18 15 12 9 6" />
                  <polyline points="15 18 9 12 15 6" transform="rotate(180, 12, 12)" />
                </svg>
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
