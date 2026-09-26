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
  className = '',
  initialPos = 50,
}) {
  const imgBefore  = beforeSrc ?? leftSrc;
  const imgAfter   = afterSrc  ?? rightSrc;
  const labelLeft  = leftLabel  ?? '◀ BEFORE / INPUT';
  const labelRight = rightLabel ?? 'AFTER / SR ▶';

  const [pos, setPos]  = useState(initialPos);   // 0–100 (%)
  const containerRef   = useRef(null);
  const isDragging     = useRef(false);

  const clamp = v => Math.min(100, Math.max(0, v));

  const updatePosFromClientX = useCallback(clientX => {
    if (!containerRef.current) return;
    const { left, width } = containerRef.current.getBoundingClientRect();
    if (width <= 0) return;
    const newPos = clamp(((clientX - left) / width) * 100);
    setPos(newPos);
  }, []);

  // ── Robust Pointer Events (Supports Mouse, Touch, Stylus) ─────────────────
  const handlePointerDown = (e) => {
    e.preventDefault();
    isDragging.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Fallback if browser doesn't support setPointerCapture
    }
    updatePosFromClientX(e.clientX);
  };

  const handlePointerMove = (e) => {
    if (isDragging.current) {
      e.preventDefault();
      updatePosFromClientX(e.clientX);
    }
  };

  const handlePointerUp = (e) => {
    if (isDragging.current) {
      isDragging.current = false;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Fallback
      }
    }
  };

  // Keyboard accessibility
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setPos(p => Math.max(0, p - 5));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setPos(p => Math.min(100, p + 5));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setPos(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setPos(100);
    }
  };

  // Global safety release
  useEffect(() => {
    const handleGlobalUp = () => { isDragging.current = false; };
    const handleGlobalMove = (e) => {
      if (isDragging.current) {
        updatePosFromClientX(e.clientX);
      }
    };
    window.addEventListener('pointerup', handleGlobalUp);
    window.addEventListener('pointermove', handleGlobalMove);
    return () => {
      window.removeEventListener('pointerup', handleGlobalUp);
      window.removeEventListener('pointermove', handleGlobalMove);
    };
  }, [updatePosFromClientX]);

  return (
    <div
      ref={containerRef}
      role="slider"
      tabIndex={0}
      aria-label="Image comparison slider"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
      onKeyDown={handleKeyDown}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className={`relative w-full h-full overflow-hidden select-none cursor-col-resize focus:outline-none focus:ring-2 focus:ring-cyan-400/50 ${className}`}
      style={{ touchAction: 'none' }}
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
