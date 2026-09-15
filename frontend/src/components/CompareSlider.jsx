/* Image Comparison Slider sub-component */
import React, { useState, useRef, useCallback } from 'react';

export default function CompareSlider({ beforeSrc, afterSrc }) {
  const [pos, setPos] = useState(50);
  const ref = useRef();
  const dragging = useRef(false);

  const move = useCallback(clientX => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    setPos(Math.min(95, Math.max(5, ((clientX - r.left) / r.width) * 100)));
  }, []);

  const onMouseDown = e => { dragging.current = true; move(e.clientX); };
  const onMouseMove = e => { if (dragging.current) move(e.clientX); };
  const onMouseUp   = ()  => { dragging.current = false; };
  const onTouchMove = e   => move(e.touches[0].clientX);

  return (
    <div
      ref={ref}
      className="viewer-wrap w-full aspect-square select-none"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onTouchMove={onTouchMove}
    >
      {/* After (SR) — full width background */}
      <img src={afterSrc} alt="SR Output" className="viewer-img absolute inset-0 w-full h-full object-cover" draggable={false} />

      {/* Before (LR) — clipped left portion */}
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <img src={beforeSrc} alt="Original LR" draggable={false}
          className="viewer-img absolute top-0 left-0 h-full object-cover"
          style={{ width: `${10000 / pos}%`, maxWidth: 'none' }} />
      </div>

      {/* Labels */}
      <span className="absolute top-3 left-3 px-2.5 py-1 text-[11px] font-bold rounded-lg"
        style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', color: 'rgba(255,255,255,0.8)' }}>
        ◀ BEFORE · 10m LR
      </span>
      <span className="absolute top-3 right-3 px-2.5 py-1 text-[11px] font-bold rounded-lg"
        style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', color: '#38bdf8' }}>
        AFTER · SR ▶
      </span>

      {/* Divider + handle */}
      <div className="absolute top-0 bottom-0 pointer-events-none" style={{ left: `${pos}%`, transform: 'translateX(-50%)', width: 2, background: 'rgba(255,255,255,0.5)' }} />
      <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full flex items-center justify-center"
        style={{ left: `${pos}%`, background: 'rgba(8,18,40,0.9)', border: '2px solid rgba(255,255,255,0.3)', backdropFilter: 'blur(10px)', pointerEvents: 'none' }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
          <polyline points="15 18 9 12 15 6" /><polyline points="9 6 15 12 9 18" />
        </svg>
      </div>
    </div>
  );
}
