/**
 * CompareSlider — production-quality drag-to-reveal image comparison
 *
 * How it works:
 *   Both images sit at IDENTICAL absolute positions (inset-0, w-full, h-full,
 *   object-cover). The "before" image is masked with CSS clip-path so only its
 *   left portion is visible. Moving the handle just changes the clip boundary —
 *   neither image moves, rescales, or re-crops. This is the same technique used
 *   by professional comparison tools (Lightroom, before-after.app, etc.).
 *
 * Props (both naming conventions accepted):
 *   beforeSrc | leftSrc   — left  / before image
 *   afterSrc  | rightSrc  — right / after  image
 *   leftLabel             — label text, left side  (optional)
 *   rightLabel            — label text, right side (optional)
 */
import React, { useState, useRef, useCallback, useEffect } from 'react';

export default function CompareSlider({
  beforeSrc, afterSrc,
  leftSrc,   rightSrc,
  leftLabel, rightLabel,
}) {
  const imgBefore  = beforeSrc ?? leftSrc;
  const imgAfter   = afterSrc  ?? rightSrc;
  const labelLeft  = leftLabel  ?? '◀ INPUT · native';
  const labelRight = rightLabel ?? 'SR OUTPUT ▶';

  const [pos, setPos]  = useState(50);   // 0–100 (%)
  const containerRef   = useRef(null);
  const dragging       = useRef(false);

  const clamp = v => Math.min(99, Math.max(1, v));

  const updatePos = useCallback(clientX => {
    if (!containerRef.current) return;
    const { left, width } = containerRef.current.getBoundingClientRect();
    setPos(clamp(((clientX - left) / width) * 100));
  }, []);

  // ── pointer events ──────────────────────────────────────────────────────
  const onMouseDown = useCallback(e => {
    e.preventDefault();
    dragging.current = true;
    updatePos(e.clientX);
  }, [updatePos]);

  const onMouseMove = useCallback(e => {
    if (dragging.current) updatePos(e.clientX);
  }, [updatePos]);

  const stopDrag = useCallback(() => { dragging.current = false; }, []);

  const onTouchStart = useCallback(e => {
    dragging.current = true;
    updatePos(e.touches[0].clientX);
  }, [updatePos]);

  const onTouchMove = useCallback(e => {
    if (dragging.current) {
      e.preventDefault();
      updatePos(e.touches[0].clientX);
    }
  }, [updatePos]);

  // release drag anywhere on the page
  useEffect(() => {
    window.addEventListener('mouseup', stopDrag);
    window.addEventListener('touchend', stopDrag);
    return () => {
      window.removeEventListener('mouseup', stopDrag);
      window.removeEventListener('touchend', stopDrag);
    };
  }, [stopDrag]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none cursor-col-resize"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseLeave={stopDrag}
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
    >
      {/* ── AFTER (right) — fills container, always fully visible ── */}
      <img
        src={imgAfter}
        alt={labelRight}
        draggable={false}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        style={{ imageRendering: 'auto' }}
      />

      {/* ── BEFORE (left) — same size, masked by clip-path ─────────────
           clip-path: inset(top right bottom left)
           right = (100 - pos)%  → hides everything to the right of the handle
           The image itself NEVER changes position or size. Only the mask moves.
      ── */}
      <img
        src={imgBefore}
        alt={labelLeft}
        draggable={false}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        style={{
          clipPath: `inset(0 ${100 - pos}% 0 0)`,
          // pixelated rendering makes the lower-res native image look visibly
          // blocky at large sizes — showing the resolution gap at a glance
          imageRendering: 'pixelated',
        }}
      />

      {/* ── Labels ────────────────────────────────────────────────────── */}
      <span
        className="absolute top-3 left-3 px-2.5 py-1 text-[11px] font-bold rounded-lg z-10 pointer-events-none"
        style={{ background: 'rgba(0,0,0,0.70)', backdropFilter: 'blur(8px)', color: 'rgba(255,255,255,0.92)' }}
      >
        {labelLeft}
      </span>
      <span
        className="absolute top-3 right-3 px-2.5 py-1 text-[11px] font-bold rounded-lg z-10 pointer-events-none"
        style={{ background: 'rgba(6,18,40,0.80)', backdropFilter: 'blur(8px)', color: '#38bdf8', border: '1px solid rgba(56,189,248,0.35)' }}
      >
        {labelRight}
      </span>

      {/* ── Divider line ──────────────────────────────────────────────── */}
      <div
        className="absolute inset-y-0 z-20 pointer-events-none"
        style={{
          left: `${pos}%`,
          transform: 'translateX(-50%)',
          width: 2,
          background: 'linear-gradient(to bottom, #ffffff 0%, #38bdf8 100%)',
          boxShadow: '0 0 10px rgba(56,189,248,0.7)',
        }}
      />

      {/* ── Drag handle ───────────────────────────────────────────────── */}
      <div
        className="absolute top-1/2 z-30 flex items-center justify-center gap-0.5 rounded-full pointer-events-none"
        style={{
          left: `${pos}%`,
          transform: 'translate(-50%, -50%)',
          width: 40,
          height: 40,
          background: 'rgba(6,18,40,0.95)',
          border: '2px solid rgba(56,189,248,0.7)',
          backdropFilter: 'blur(12px)',
          boxShadow: '0 0 20px rgba(56,189,248,0.5)',
        }}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </div>
    </div>
  );
}
