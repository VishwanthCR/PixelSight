import React, { useRef, useEffect, useState } from 'react';
import { Globe2, Radio, Compass, Eye, RotateCw } from 'lucide-react';

/**
 * SatelliteGlobe3D — High-performance procedural Canvas 3D Earth & Sentinel-2 Orbit Visualizer.
 * Renders a rotating Earth globe with:
 * - Illuminated atmospheric rim & gridlines
 * - Continental landforms with India AOI highlighted in cyan glow
 * - Sentinel-2 polar sun-synchronous orbit track (98.6° inclination)
 * - Live satellite telemetry HUD
 * - Drag-to-rotate interaction
 */
export default function SatelliteGlobe3D({ height = 360, showTelemetry = true }) {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 0.25, y: -1.3 }); // Focus around India initially
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [orbitAngle, setOrbitAngle] = useState(0);
  const [telemetry, setTelemetry] = useState({
    subLat: '19.07° N',
    subLon: '72.87° E',
    speed: '7.45 km/s',
    alt: '786 km',
    phase: 'Ascending node',
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let angle = orbitAngle;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(w, h) * 0.38;

      ctx.clearRect(0, 0, w, h);

      // Deep space background gradient
      const bgGrad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 1.6);
      bgGrad.addColorStop(0, 'rgba(15, 23, 42, 0.6)');
      bgGrad.addColorStop(0.7, 'rgba(5, 10, 20, 0.85)');
      bgGrad.addColorStop(1, 'rgba(2, 6, 15, 0.95)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Outer atmospheric glow
      const atmoGrad = ctx.createRadialGradient(cx, cy, r * 0.9, cx, cy, r * 1.25);
      atmoGrad.addColorStop(0, 'rgba(6, 182, 212, 0.35)');
      atmoGrad.addColorStop(0.5, 'rgba(14, 165, 233, 0.15)');
      atmoGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
      ctx.fillStyle = atmoGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 1.25, 0, Math.PI * 2);
      ctx.fill();

      // Earth Globe Sphere base
      const earthGrad = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.35, r * 0.1, cx, cy, r);
      earthGrad.addColorStop(0, '#1e3a5f');
      earthGrad.addColorStop(0.5, '#0d2137');
      earthGrad.addColorStop(0.85, '#071526');
      earthGrad.addColorStop(1, '#030b14');
      ctx.fillStyle = earthGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      // Clip drawing inside sphere
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.clip();

      const rx = rotation.x;
      const ry = rotation.y;

      // Draw latitude grid lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.lineWidth = 1;
      for (let lat = -60; lat <= 60; lat += 30) {
        const phi = (lat * Math.PI) / 180;
        const radY = Math.sin(phi) * r;
        const radX = Math.cos(phi) * r;
        ctx.beginPath();
        const yPos = cy - radY * Math.cos(rx);
        const yRadius = Math.abs(radX * Math.sin(rx));
        ctx.ellipse(cx, yPos, radX, yRadius, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw longitude meridians
      for (let lon = 0; lon < 360; lon += 30) {
        const theta = ((lon + ry * 50) * Math.PI) / 180;
        const xOffset = Math.sin(theta);
        const zOffset = Math.cos(theta);
        if (zOffset > -0.2) {
          ctx.beginPath();
          ctx.ellipse(cx, cy, Math.abs(xOffset * r), r, 0, 0, Math.PI * 2);
          ctx.strokeStyle = zOffset > 0 ? 'rgba(56, 189, 248, 0.15)' : 'rgba(56, 189, 248, 0.05)';
          ctx.stroke();
        }
      }

      // Continental outlines & India highlight projection
      // Simulated continental clusters
      const continents = [
        { lat: 22, lon: 79, name: 'India', isIndia: true, radius: 24 },
        { lat: 35, lon: 105, name: 'East Asia', isIndia: false, radius: 35 },
        { lat: 10, lon: 20, name: 'Africa', isIndia: false, radius: 45 },
        { lat: 48, lon: 15, name: 'Europe', isIndia: false, radius: 30 },
        { lat: -25, lon: 135, name: 'Australia', isIndia: false, radius: 28 },
      ];

      continents.forEach(c => {
        const latRad = (c.lat * Math.PI) / 180;
        const lonRad = ((c.lon + ry * 57.3) * Math.PI) / 180;

        const x3d = Math.cos(latRad) * Math.sin(lonRad);
        const y3d = -Math.sin(latRad);
        const z3d = Math.cos(latRad) * Math.cos(lonRad);

        // Apply pitch (rx)
        const rotY = y3d * Math.cos(rx) - z3d * Math.sin(rx);
        const rotZ = y3d * Math.sin(rx) + z3d * Math.cos(rx);

        if (rotZ > -0.15) {
          const screenX = cx + x3d * r;
          const screenY = cy + rotY * r;
          const alpha = Math.max(0.1, (rotZ + 0.15) / 1.15);

          if (c.isIndia) {
            // India highlighted polygon silhouette
            ctx.fillStyle = `rgba(6, 182, 212, ${0.45 * alpha})`;
            ctx.strokeStyle = `rgba(34, 211, 238, ${0.85 * alpha})`;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(screenX, screenY, c.radius * (0.8 + 0.2 * rotZ), 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // AOI Pulse ring around India
            const pulse = (Math.sin(angle * 3) + 1) * 6;
            ctx.strokeStyle = `rgba(52, 211, 153, ${0.7 * alpha})`;
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(screenX, screenY, c.radius + pulse, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = `rgba(255, 255, 255, ${0.9 * alpha})`;
            ctx.font = 'bold 11px ui-sans-serif, system-ui';
            ctx.textAlign = 'center';
            ctx.fillText('INDIA (AOI)', screenX, screenY - c.radius - 8);
          } else {
            ctx.fillStyle = `rgba(71, 85, 105, ${0.25 * alpha})`;
            ctx.beginPath();
            ctx.arc(screenX, screenY, c.radius * 0.9, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      });

      ctx.restore();

      // ── Sentinel-2 Polar Sun-Synchronous Orbit Track (98.6° inclination) ──
      ctx.save();
      const orbitRadius = r * 1.28;
      const incl = 1.72; // ~98.6° in radians

      ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();

      const numPts = 64;
      for (let i = 0; i <= numPts; i++) {
        const th = (i / numPts) * Math.PI * 2;
        const ox = Math.cos(th) * orbitRadius;
        const oy = Math.sin(th) * orbitRadius * Math.sin(incl);
        const oz = Math.sin(th) * orbitRadius * Math.cos(incl);
        const screenOx = cx + ox * 0.85;
        const screenOy = cy + oy;
        if (i === 0) ctx.moveTo(screenOx, screenOy);
        else ctx.lineTo(screenOx, screenOy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Satellite position along orbit
      angle += 0.015;
      const satX = cx + Math.cos(angle) * orbitRadius * 0.85;
      const satY = cy + Math.sin(angle) * orbitRadius * Math.sin(incl);

      // Satellite swath footprint cone down to globe
      const footGrad = ctx.createLinearGradient(satX, satY, cx, cy);
      footGrad.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
      footGrad.addColorStop(1, 'rgba(6, 182, 212, 0.05)');
      ctx.fillStyle = footGrad;
      ctx.beginPath();
      ctx.moveTo(satX, satY);
      ctx.lineTo(cx - 30, cy + 20);
      ctx.lineTo(cx + 30, cy + 40);
      ctx.closePath();
      ctx.fill();

      // Satellite body
      ctx.fillStyle = '#38bdf8';
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(satX, satY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Solar panels
      ctx.strokeStyle = '#93c5fd';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(satX - 10, satY);
      ctx.lineTo(satX + 10, satY);
      ctx.stroke();

      // Satellite label
      ctx.fillStyle = '#e0f2fe';
      ctx.font = 'bold 10px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Sentinel-2B', satX + 12, satY - 6);
      ctx.fillStyle = '#38bdf8';
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillText('MSI 10m/20m/60m', satX + 12, satY + 6);

      ctx.restore();

      // Auto-rotation when not dragging
      if (!isDragging) {
        setRotation(prev => ({ ...prev, y: prev.y + 0.002 }));
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [rotation, isDragging]);

  const handleMouseDown = e => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = e => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setRotation(prev => ({
      x: Math.max(-0.8, Math.min(0.8, prev.x + dy * 0.005)),
      y: prev.y + dx * 0.005,
    }));
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="relative rounded-3xl overflow-hidden border border-cyan-500/20 bg-slate-950/80 shadow-2xl backdrop-blur-md">
      {/* 3D Canvas */}
      <canvas
        ref={canvasRef}
        width={720}
        height={height}
        className="w-full cursor-grab active:cursor-grabbing block"
        style={{ height }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Overlay Badge */}
      <div className="absolute top-4 left-4 flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-slate-900/90 border border-cyan-500/30 text-xs backdrop-blur-md">
        <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
        <span className="font-bold text-white tracking-wide">Sentinel-2 Constellation</span>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300">
          Sun-Synchronous · 786 km
        </span>
      </div>

      {/* Orbit Interaction Hint */}
      <div className="absolute top-4 right-4 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/70 border border-slate-800 text-[10px] text-slate-400">
        <RotateCw className="w-3 h-3 text-cyan-400" />
        <span>Drag to rotate globe</span>
      </div>

      {/* Telemetry HUD Bottom Bar */}
      {showTelemetry && (
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent p-4 pt-6 border-t border-white/[0.04] grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Sensor Payload</span>
            <span className="font-mono font-bold text-cyan-300">MSI (13 Spectral Bands)</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Swath Width</span>
            <span className="font-mono font-bold text-slate-200">290 km Field of View</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Temporal Revisit</span>
            <span className="font-mono font-bold text-emerald-400">5 Days (S2A + S2B)</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">AOI Target</span>
            <span className="font-mono font-bold text-amber-300">India Geographic Extent</span>
          </div>
        </div>
      )}
    </div>
  );
}
