import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers, MapPin, CheckCircle2, AlertTriangle, ShieldCheck,
  RotateCcw, Redo2, Undo2, Trash2, Save, Download, Play,
  Eye, Grid, HelpCircle, User, Check, ArrowLeft, FileJson,
  FileSpreadsheet, Sparkles, AlertCircle, Info, ChevronRight,
  Crosshair, Square, Move
} from 'lucide-react';
import {
  fetchGroundTruth,
  saveGroundTruthAnnotations,
  validateGroundTruth,
  rasterizeGroundTruth,
  reviewGroundTruth,
  groundTruthExportUrl,
  resultFileUrl,
} from '../api/srmApi.js';

// 8 Canonical Land Cover Classes according to prompt Section 2
export const GT_CLASSES = [
  { id: 0, name: 'Tree', color: '#28b45a', desc: 'Trees & closed forest canopy' },
  { id: 1, name: 'Shrubland', color: '#78aa50', desc: 'Shrub and bush formations' },
  { id: 2, name: 'Grassland', color: '#aad264', desc: 'Natural & semi-natural herbaceous' },
  { id: 3, name: 'Cropland', color: '#dcbe46', desc: 'Cultivated agricultural fields' },
  { id: 4, name: 'Built-up', color: '#d25a37', desc: 'Impervious structures & building clusters' },
  { id: 5, name: 'Bare', color: '#96876e', desc: 'Bare soil, sand, and rock surfaces' },
  { id: 6, name: 'Water', color: '#327dd2', desc: 'Permanent and seasonal open water' },
  { id: 255, name: 'Ignore', color: '#64748b', desc: 'No-data / unclassified background mask' },
];

export default function GroundTruthPage({ initialJobId = 'demo-urban-001', onBack }) {
  const [jobId, setJobId] = useState(initialJobId);
  const [activeJobId, setActiveJobId] = useState(initialJobId);

  // Annotation Tool State
  const [selectedClassId, setSelectedClassId] = useState(0);
  const [activeTool, setActiveTool] = useState('polygon'); // 'polygon' | 'brush' | 'eraser' | 'select'
  const [annotatorName, setAnnotatorName] = useState('Senior RS Analyst');
  const [reviewerName, setReviewerName] = useState('Principal RS Scientist');
  const [evaluationGrid, setEvaluationGrid] = useState('2.5m'); // '2.5m' | '10m'

  // Canvas / Drawing State (Coordinates in normalized 0..1 space or 512x512 image space)
  const CANVAS_SIZE = 512;
  const [polygons, setPolygons] = useState([]);
  const [currentPoints, setCurrentPoints] = useState([]);
  const [selectedPolygonIdx, setSelectedPolygonIdx] = useState(null);
  const [brushStart, setBrushStart] = useState(null);

  // History for Undo / Redo
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Backend GT State
  const [gtData, setGtData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validating, setValidating] = useState(false);
  const [rasterizing, setRasterizing] = useState(false);
  const [validationReport, setValidationReport] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);

  // Backdrop selection
  const [backdrop, setBackdrop] = useState('hr'); // 'hr' | 'native' | 'dark'

  const canvasRef = useRef(null);

  // Push new state to history
  const pushHistory = useCallback((newPolygons) => {
    setHistory(prev => {
      const upToCurrent = prev.slice(0, historyIndex + 1);
      return [...upToCurrent, newPolygons];
    });
    setHistoryIndex(prev => prev + 1);
  }, [historyIndex]);

  // Load GT data from backend
  const loadWorkspace = useCallback(async (targetJobId) => {
    if (!targetJobId) return;
    setLoading(true);
    setStatusMessage(null);
    try {
      const data = await fetchGroundTruth(targetJobId);
      setGtData(data);
      if (data.metadata?.annotator) setAnnotatorName(data.metadata.annotator);
      if (data.metadata?.reviewer) setReviewerName(data.metadata.reviewer);
      if (data.metadata?.evaluation_grid) setEvaluationGrid(data.metadata.evaluation_grid);

      // Convert GeoJSON features into canvas polygon coordinate array
      if (data.annotations?.features?.length > 0) {
        const loaded = data.annotations.features.map((f, i) => {
          const ring = f.geometry?.coordinates?.[0] || [];
          // If coordinates look like pixel points [x, y], use directly; if normalized, scale
          const pts = ring.slice(0, -1).map(([x, y]) => ({ x, y }));
          return {
            id: `poly_${i}_${Date.now()}`,
            class_id: f.properties?.class_id ?? 0,
            class_name: f.properties?.class_name ?? 'Tree',
            points: pts,
          };
        });
        setPolygons(loaded);
        setHistory([loaded]);
        setHistoryIndex(0);
      } else {
        setPolygons([]);
        setHistory([[]]);
        setHistoryIndex(0);
      }
    } catch (err) {
      // If not exists yet, we initialize a clean draft
      setPolygons([]);
      setHistory([[]]);
      setHistoryIndex(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadWorkspace(activeJobId);
  }, [activeJobId, loadWorkspace]);

  // Undo / Redo handlers
  const handleUndo = () => {
    if (historyIndex > 0) {
      const prevIdx = historyIndex - 1;
      setHistoryIndex(prevIdx);
      setPolygons(history[prevIdx]);
      setSelectedPolygonIdx(null);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const nextIdx = historyIndex + 1;
      setHistoryIndex(nextIdx);
      setPolygons(history[nextIdx]);
      setSelectedPolygonIdx(null);
    }
  };

  // Canvas Mouse Coordinates helper
  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    const x = Math.max(0, Math.min(CANVAS_SIZE, Math.round((e.clientX - rect.left) * scaleX)));
    const y = Math.max(0, Math.min(CANVAS_SIZE, Math.round((e.clientY - rect.top) * scaleY)));
    return { x, y };
  };

  // Canvas Click / Mouse Interactions
  const handleCanvasClick = (e) => {
    const pt = getCanvasCoords(e);

    if (activeTool === 'polygon') {
      // If clicking near first point (> 2 points), close polygon
      if (currentPoints.length >= 3) {
        const first = currentPoints[0];
        const dist = Math.hypot(pt.x - first.x, pt.y - first.y);
        if (dist < 15) {
          finishPolygon();
          return;
        }
      }
      setCurrentPoints(prev => [...prev, pt]);
    } else if (activeTool === 'eraser') {
      // Find top polygon under cursor and delete it
      const clickedIdx = findPolygonUnderCursor(pt);
      if (clickedIdx !== null) {
        const nextPolys = polygons.filter((_, idx) => idx !== clickedIdx);
        setPolygons(nextPolys);
        pushHistory(nextPolys);
        setSelectedPolygonIdx(null);
      }
    } else if (activeTool === 'select') {
      const clickedIdx = findPolygonUnderCursor(pt);
      setSelectedPolygonIdx(clickedIdx);
    }
  };

  // Brush / Box Drawing handlers
  const handleMouseDown = (e) => {
    if (activeTool === 'brush') {
      const pt = getCanvasCoords(e);
      setBrushStart(pt);
    }
  };

  const handleMouseUp = (e) => {
    if (activeTool === 'brush' && brushStart) {
      const pt = getCanvasCoords(e);
      const minX = Math.min(brushStart.x, pt.x);
      const maxX = Math.max(brushStart.x, pt.x);
      const minY = Math.min(brushStart.y, pt.y);
      const maxY = Math.max(brushStart.y, pt.y);

      // Only add if reasonable area
      if (maxX - minX > 5 && maxY - minY > 5) {
        const targetClass = GT_CLASSES.find(c => c.id === selectedClassId) || GT_CLASSES[0];
        const boxPoly = {
          id: `brush_${Date.now()}`,
          class_id: targetClass.id,
          class_name: targetClass.name,
          points: [
            { x: minX, y: minY },
            { x: maxX, y: minY },
            { x: maxX, y: maxY },
            { x: minX, y: maxY },
          ],
        };
        const nextPolys = [...polygons, boxPoly];
        setPolygons(nextPolys);
        pushHistory(nextPolys);
      }
      setBrushStart(null);
    }
  };

  // Complete current polygon
  const finishPolygon = () => {
    if (currentPoints.length < 3) {
      setCurrentPoints([]);
      return;
    }
    const targetClass = GT_CLASSES.find(c => c.id === selectedClassId) || GT_CLASSES[0];
    const newPoly = {
      id: `poly_${Date.now()}`,
      class_id: targetClass.id,
      class_name: targetClass.name,
      points: [...currentPoints],
    };
    const nextPolys = [...polygons, newPoly];
    setPolygons(nextPolys);
    pushHistory(nextPolys);
    setCurrentPoints([]);
  };

  // Double click finishes polygon
  const handleDoubleClick = () => {
    if (activeTool === 'polygon') {
      finishPolygon();
    }
  };

  const handleDeleteSelected = () => {
    if (selectedPolygonIdx !== null) {
      const nextPolys = polygons.filter((_, idx) => idx !== selectedPolygonIdx);
      setPolygons(nextPolys);
      pushHistory(nextPolys);
      setSelectedPolygonIdx(null);
    }
  };

  const handleClearAll = () => {
    if (window.confirm('Clear all annotated polygons for this AOI?')) {
      setPolygons([]);
      setCurrentPoints([]);
      setSelectedPolygonIdx(null);
      pushHistory([]);
    }
  };

  // Simple point-in-polygon algorithm for eraser and selection
  const findPolygonUnderCursor = (pt) => {
    for (let i = polygons.length - 1; i >= 0; i--) {
      const poly = polygons[i];
      let inside = false;
      const pts = poly.points;
      for (let j = 0, k = pts.length - 1; j < pts.length; k = j++) {
        const xi = pts[j].x, yi = pts[j].y;
        const xj = pts[k].x, yj = pts[k].y;
        const intersect = ((yi > pt.y) !== (yj > pt.y)) && (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
      }
      if (inside) return i;
    }
    return null;
  };

  // Build GeoJSON from current state
  const buildGeoJSON = () => {
    const features = polygons.map(p => {
      // Closed loop coordinates [ [x1,y1], [x2,y2], ..., [x1,y1] ]
      const coords = p.points.map(pt => [pt.x, pt.y]);
      if (coords.length > 0) {
        coords.push([coords[0][0], coords[0][1]]);
      }
      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [coords],
        },
        properties: {
          class_id: p.class_id,
          class_name: p.class_name,
          annotator: annotatorName,
          created_at: new Date().toISOString(),
        },
      };
    });

    return {
      type: 'FeatureCollection',
      crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
      features,
    };
  };

  // Save draft annotations
  const handleSave = async () => {
    setSaving(true);
    setStatusMessage(null);
    try {
      const geojson = buildGeoJSON();
      const res = await saveGroundTruthAnnotations(activeJobId, geojson, annotatorName, 'Vector expert annotation');
      setGtData(res.ground_truth || res);
      setStatusMessage({ type: 'success', text: `Saved ${polygons.length} vector polygons.` });
    } catch (err) {
      setStatusMessage({ type: 'error', text: `Failed to save: ${err.message}` });
    } finally {
      setSaving(false);
    }
  };

  // Validate annotations
  const handleValidate = async () => {
    setValidating(true);
    setStatusMessage(null);
    try {
      // Save current state first to ensure backend has latest
      const geojson = buildGeoJSON();
      await saveGroundTruthAnnotations(activeJobId, geojson, annotatorName);
      const res = await validateGroundTruth(activeJobId);
      setValidationReport(res);
      if (res.is_valid) {
        setStatusMessage({ type: 'success', text: 'Annotations passed geometry and semantic validation!' });
      } else {
        setStatusMessage({ type: 'warning', text: `Validation found ${res.errors?.length || 0} issues.` });
      }
    } catch (err) {
      setStatusMessage({ type: 'error', text: `Validation failed: ${err.message}` });
    } finally {
      setValidating(false);
    }
  };

  // Transition Review Status
  const handleReviewStatus = async (newStatus) => {
    setLoading(true);
    setStatusMessage(null);
    try {
      const res = await reviewGroundTruth(activeJobId, newStatus, reviewerName);
      setGtData(res.ground_truth || res);
      setStatusMessage({
        type: 'success',
        text: `Status updated to ${newStatus.toUpperCase()}${newStatus === 'validated' ? ' (Eligible as Ground Truth)' : ''}.`,
      });
    } catch (err) {
      setStatusMessage({ type: 'error', text: `Review update failed: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  // Rasterize onto Common Evaluation Grid
  const handleRasterize = async () => {
    setRasterizing(true);
    setStatusMessage(null);
    try {
      const res = await rasterizeGroundTruth(activeJobId, evaluationGrid, true);
      setGtData(res.ground_truth || res);
      setStatusMessage({
        type: 'success',
        text: `Rasterization complete on ${evaluationGrid} grid. Coverage: ${res.raster_stats?.coverage_percentage ?? 100}%.`,
      });
      // Refresh workspace data
      loadWorkspace(activeJobId);
    } catch (err) {
      setStatusMessage({ type: 'error', text: `Rasterization failed: ${err.message}` });
    } finally {
      setRasterizing(false);
    }
  };

  // Backdrop image URLs
  const hrBackdropUrl = resultFileUrl(activeJobId, 'hr_reference_aligned.png');
  const nativeBackdropUrl = resultFileUrl(activeJobId, 'previews/original_rgb.png');

  const currentStatus = gtData?.metadata?.validation_status || 'draft';

  return (
    <div className="min-h-screen bg-[#030712] text-slate-200 pb-16">
      {/* ──────────────────────────────────────────────────────────── */}
      {/* 1. Header & Navigation                                       */}
      {/* ──────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 bg-slate-950/90 backdrop-blur-md border-b border-white/[0.08] px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {onBack && (
              <button
                onClick={onBack}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white bg-slate-900 border border-slate-800 transition-colors"
                title="Back to Dashboard"
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
            )}
            <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 shadow-sm">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-white tracking-tight">GROUND TRUTH ANNOTATION WORKSPACE</h1>
                <span className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full border font-semibold ${
                  currentStatus === 'validated'
                    ? 'border-emerald-500/40 text-emerald-300 bg-emerald-500/15'
                    : currentStatus === 'review'
                    ? 'border-blue-500/40 text-blue-300 bg-blue-500/15'
                    : 'border-amber-500/40 text-amber-300 bg-amber-500/15'
                }`}>
                  {currentStatus === 'validated' ? '✓ GROUND_TRUTH' : `Status: ${currentStatus}`}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Annotate reference polygons over HR imagery for independent cross-model benchmark evaluation
              </p>
            </div>
          </div>

          {/* AOI Selector & Job Input */}
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-900 px-3 py-1.5 rounded-xl border border-white/[0.08] text-xs">
              <span className="text-slate-500 font-mono mr-2">AOI / Job:</span>
              <input
                type="text"
                value={jobId}
                onChange={e => setJobId(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') setActiveJobId(jobId); }}
                className="bg-transparent font-mono text-slate-200 outline-none w-36"
                placeholder="job-id"
              />
              <button
                onClick={() => setActiveJobId(jobId)}
                className="text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 ml-2"
              >
                Load
              </button>
            </div>

            {/* Quick Demo Pre-load buttons */}
            <div className="hidden lg:flex items-center gap-1 text-[11px] font-mono">
              {['demo-urban-001', 'demo-crop-001', 'demo-001'].map(id => (
                <button
                  key={id}
                  onClick={() => { setJobId(id); setActiveJobId(id); }}
                  className={`px-2 py-1 rounded-lg border transition-all ${
                    activeJobId === id
                      ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                      : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-white'
                  }`}
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 2. Notification banner                                       */}
      {/* ──────────────────────────────────────────────────────────── */}
      {statusMessage && (
        <div className={`px-6 py-2.5 text-xs flex items-center justify-between border-b ${
          statusMessage.type === 'success'
            ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-200'
            : statusMessage.type === 'warning'
            ? 'bg-amber-950/40 border-amber-500/30 text-amber-200'
            : 'bg-red-950/40 border-red-500/30 text-red-200'
        }`}>
          <div className="max-w-7xl mx-auto w-full flex items-center gap-2">
            {statusMessage.type === 'success' ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <AlertTriangle className="w-4 h-4 text-amber-400" />}
            <span>{statusMessage.text}</span>
          </div>
          <button onClick={() => setStatusMessage(null)} className="text-slate-400 hover:text-white text-sm">✕</button>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-6 pt-6 space-y-6">
        {/* ──────────────────────────────────────────────────────────── */}
        {/* 3. Toolbar & Class Palette (Section 2)                       */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="card p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-3">
            {/* Tool Selection */}
            <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => { setActiveTool('polygon'); setCurrentPoints([]); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTool === 'polygon'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Polygon tool: Click points to draw, close at first point"
              >
                <Crosshair className="w-3.5 h-3.5" />
                <span>Polygon</span>
              </button>

              <button
                onClick={() => { setActiveTool('brush'); setCurrentPoints([]); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTool === 'brush'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Brush/Box tool: Click and drag to create rectangular label"
              >
                <Square className="w-3.5 h-3.5" />
                <span>Box / Brush</span>
              </button>

              <button
                onClick={() => { setActiveTool('select'); setCurrentPoints([]); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTool === 'select'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Select polygon to inspect or delete"
              >
                <Move className="w-3.5 h-3.5" />
                <span>Select</span>
              </button>

              <button
                onClick={() => { setActiveTool('eraser'); setCurrentPoints([]); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTool === 'eraser'
                    ? 'bg-red-500/20 text-red-300 border border-red-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Eraser tool: Click any polygon to delete"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Eraser</span>
              </button>
            </div>

            {/* History & Delete Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleUndo}
                disabled={historyIndex <= 0}
                className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white disabled:opacity-40 disabled:hover:text-slate-300 transition-colors"
                title="Undo (Ctrl+Z)"
              >
                <Undo2 className="w-4 h-4" />
              </button>
              <button
                onClick={handleRedo}
                disabled={historyIndex >= history.length - 1}
                className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white disabled:opacity-40 disabled:hover:text-slate-300 transition-colors"
                title="Redo (Ctrl+Y)"
              >
                <Redo2 className="w-4 h-4" />
              </button>
              {selectedPolygonIdx !== null && (
                <button
                  onClick={handleDeleteSelected}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs font-semibold hover:bg-red-500/20"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Delete Selected</span>
                </button>
              )}
              <button
                onClick={handleClearAll}
                className="text-xs text-slate-500 hover:text-red-400 px-2 py-1 rounded transition-colors"
              >
                Clear All
              </button>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-200 hover:bg-slate-700 text-xs font-semibold shadow-sm transition-all"
              >
                <Save className="w-3.5 h-3.5 text-cyan-400" />
                <span>{saving ? 'Saving...' : 'Save Draft'}</span>
              </button>

              <button
                onClick={handleValidate}
                disabled={validating}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500/15 border border-blue-500/30 text-blue-300 hover:bg-blue-500/25 text-xs font-semibold shadow-sm transition-all"
              >
                <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
                <span>{validating ? 'Validating...' : 'Validate'}</span>
              </button>

              <button
                onClick={handleRasterize}
                disabled={rasterizing || polygons.length === 0}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-bold shadow-md shadow-emerald-900/30 transition-all disabled:opacity-40"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{rasterizing ? 'Rasterizing...' : 'Rasterize onto Common Grid'}</span>
              </button>
            </div>
          </div>

          {/* 8 Canonical Land Cover Classes */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Land Cover Class Palette (Section 2)
              </span>
              <span className="text-[11px] text-slate-500 font-mono">
                {polygons.length} polygons total
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
              {GT_CLASSES.map(cls => {
                const isSelected = selectedClassId === cls.id;
                const count = polygons.filter(p => p.class_id === cls.id).length;
                return (
                  <button
                    key={cls.id}
                    onClick={() => setSelectedClassId(cls.id)}
                    className={`flex flex-col p-2 rounded-xl text-left border transition-all ${
                      isSelected
                        ? 'border-white/40 ring-2 shadow-lg bg-slate-900/90'
                        : 'border-white/[0.05] bg-slate-950/60 hover:bg-slate-900/50'
                    }`}
                    style={isSelected ? { ringColor: cls.color } : {}}
                  >
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: cls.color }} />
                      <span className="text-[10px] font-mono px-1 rounded bg-black/40 text-slate-400">
                        {count}
                      </span>
                    </div>
                    <span className="text-xs font-bold text-slate-200 truncate">{cls.name}</span>
                    <span className="text-[9px] text-slate-500 truncate">{cls.desc}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* ──────────────────────────────────────────────────────────── */}
        {/* 4. Interactive Workspace & Preview Window (Section 2)        */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="grid lg:grid-cols-12 gap-6">
          {/* Main Canvas Area */}
          <div className="lg:col-span-8 card p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] pb-2 text-xs">
              <div className="flex items-center gap-3">
                <span className="text-slate-400 font-medium">Backdrop Image:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800">
                  <button
                    onClick={() => setBackdrop('hr')}
                    className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                      backdrop === 'hr' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    HR Reference (2.5m)
                  </button>
                  <button
                    onClick={() => setBackdrop('native')}
                    className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                      backdrop === 'native' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Sentinel-2 (10m)
                  </button>
                  <button
                    onClick={() => setBackdrop('dark')}
                    className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                      backdrop === 'dark' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Dark Grid
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400">
                <span>Tool: <strong>{activeTool.toUpperCase()}</strong></span>
                {activeTool === 'polygon' && (
                  <span className="text-cyan-400">
                    {currentPoints.length === 0 ? 'Click to start polygon' : `${currentPoints.length} vertices (click start point or double-click to close)`}
                  </span>
                )}
              </div>
            </div>

            {/* Canvas Container */}
            <div className="relative aspect-square max-w-[540px] mx-auto rounded-xl overflow-hidden border border-white/[0.1] bg-[#0a0f1d] shadow-2xl select-none">
              {/* Image Backdrop */}
              {backdrop === 'hr' && (
                <img
                  src={hrBackdropUrl}
                  alt="High-Resolution Reference Backdrop"
                  className="absolute inset-0 w-full h-full object-cover pointer-events-none opacity-80"
                  onError={e => { e.target.style.display = 'none'; }}
                />
              )}
              {backdrop === 'native' && (
                <img
                  src={nativeBackdropUrl}
                  alt="Native Sentinel-2 Backdrop"
                  className="absolute inset-0 w-full h-full object-cover pointer-events-none opacity-80"
                  onError={e => { e.target.style.display = 'none'; }}
                />
              )}

              {/* Grid overlay for solid background */}
              {backdrop === 'dark' && (
                <div
                  className="absolute inset-0 pointer-events-none opacity-20"
                  style={{
                    backgroundImage: 'linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)',
                    backgroundSize: '32px 32px'
                  }}
                />
              )}

              {/* SVG Vector Overlay */}
              <svg
                viewBox={`0 0 ${CANVAS_SIZE} ${CANVAS_SIZE}`}
                className="absolute inset-0 w-full h-full pointer-events-none"
              >
                {/* Committed Polygons */}
                {polygons.map((p, idx) => {
                  const ptsString = p.points.map(pt => `${pt.x},${pt.y}`).join(' ');
                  const cls = GT_CLASSES.find(c => c.id === p.class_id) || GT_CLASSES[0];
                  const isSelected = selectedPolygonIdx === idx;
                  return (
                    <g key={p.id || idx}>
                      <polygon
                        points={ptsString}
                        fill={cls.color}
                        fillOpacity={isSelected ? 0.65 : 0.45}
                        stroke={isSelected ? '#ffffff' : cls.color}
                        strokeWidth={isSelected ? 3 : 1.5}
                        strokeDasharray={isSelected ? '4,4' : 'none'}
                      />
                      {/* Vertex handles if selected */}
                      {isSelected && p.points.map((pt, vIdx) => (
                        <circle
                          key={vIdx}
                          cx={pt.x}
                          cy={pt.y}
                          r={3.5}
                          fill="#ffffff"
                          stroke={cls.color}
                          strokeWidth={1.5}
                        />
                      ))}
                    </g>
                  );
                })}

                {/* Currently In-Progress Polygon */}
                {currentPoints.length > 0 && (
                  <g>
                    {currentPoints.length > 1 && (
                      <polyline
                        points={currentPoints.map(pt => `${pt.x},${pt.y}`).join(' ')}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={2}
                        strokeDasharray="4,4"
                      />
                    )}
                    {currentPoints.map((pt, i) => (
                      <circle
                        key={i}
                        cx={pt.x}
                        cy={pt.y}
                        r={i === 0 ? 5 : 3.5}
                        fill={i === 0 ? '#38bdf8' : '#ffffff'}
                        stroke="#000000"
                        strokeWidth={1}
                      />
                    ))}
                  </g>
                )}

                {/* Active Box Brush Drag Preview */}
                {brushStart && (
                  <rect
                    x={brushStart.x}
                    y={brushStart.y}
                    width={10}
                    height={10}
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth={1.5}
                  />
                )}
              </svg>

              {/* Interaction Canvas Layer */}
              <div
                ref={canvasRef}
                onClick={handleCanvasClick}
                onDoubleClick={handleDoubleClick}
                onMouseDown={handleMouseDown}
                onMouseUp={handleMouseUp}
                className="absolute inset-0 w-full h-full cursor-crosshair"
              />
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
              <span>Grid: 512 × 512 normalized space</span>
              <span>Double-click or click initial point to close polygon</span>
            </div>
          </div>

          {/* Right Sidebar: Review Workflow, Evaluation Grid, & Exports */}
          <div className="lg:col-span-4 space-y-6">
            {/* 1. Review Workflow (Section 6) */}
            <div className="card p-5 space-y-4 border border-white/[0.08]">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                    Review Workflow
                  </h3>
                </div>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full uppercase font-bold ${
                  currentStatus === 'validated'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : currentStatus === 'review'
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                    : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                }`}>
                  {currentStatus}
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="text-[10px] text-slate-400 block mb-1 uppercase font-mono">Annotator</label>
                  <input
                    type="text"
                    value={annotatorName}
                    onChange={e => setAnnotatorName(e.target.value)}
                    className="w-full bg-slate-900 border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
                  />
                </div>

                <div>
                  <label className="text-[10px] text-slate-400 block mb-1 uppercase font-mono">Reviewer</label>
                  <input
                    type="text"
                    value={reviewerName}
                    onChange={e => setReviewerName(e.target.value)}
                    className="w-full bg-slate-900 border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
                  />
                </div>

                {/* Status Transitions */}
                <div className="pt-2 space-y-2">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Status Actions:</span>
                  <div className="grid grid-cols-3 gap-1.5">
                    <button
                      onClick={() => handleReviewStatus('draft')}
                      className={`px-2 py-1.5 rounded-lg border text-center transition-all ${
                        currentStatus === 'draft'
                          ? 'border-amber-500/50 bg-amber-500/20 text-amber-300 font-bold'
                          : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-white'
                      }`}
                    >
                      Draft
                    </button>
                    <button
                      onClick={() => handleReviewStatus('review')}
                      className={`px-2 py-1.5 rounded-lg border text-center transition-all ${
                        currentStatus === 'review'
                          ? 'border-blue-500/50 bg-blue-500/20 text-blue-300 font-bold'
                          : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-white'
                      }`}
                    >
                      In Review
                    </button>
                    <button
                      onClick={() => handleReviewStatus('validated')}
                      className={`px-2 py-1.5 rounded-lg border text-center transition-all ${
                        currentStatus === 'validated'
                          ? 'border-emerald-500/50 bg-emerald-500/20 text-emerald-300 font-bold'
                          : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-white'
                      }`}
                    >
                      Validate
                    </button>
                  </div>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-white/[0.04] text-[11px] text-slate-400 leading-relaxed">
                  <strong>Scientific Rule:</strong> Only annotations set to <code>validated</code> are treated as <code>GROUND_TRUTH</code> in the classification benchmarks. Draft annotations remain unbenchmarked.
                </div>
              </div>
            </div>

            {/* 2. Common Evaluation Grid (Section 7) */}
            <div className="card p-5 space-y-4 border border-white/[0.08]">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                <div className="flex items-center gap-2">
                  <Grid className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                    Common Evaluation Grid
                  </h3>
                </div>
              </div>

              <div className="space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setEvaluationGrid('2.5m')}
                    className={`p-2.5 rounded-xl border text-left transition-all ${
                      evaluationGrid === '2.5m'
                        ? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-300 font-bold'
                        : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-white'
                    }`}
                  >
                    <span className="block text-xs">2.5 m (SR Grid)</span>
                    <span className="text-[10px] opacity-75 font-normal">512 × 512 px raster</span>
                  </button>

                  <button
                    onClick={() => setEvaluationGrid('10m')}
                    className={`p-2.5 rounded-xl border text-left transition-all ${
                      evaluationGrid === '10m'
                        ? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-300 font-bold'
                        : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-white'
                    }`}
                  >
                    <span className="block text-xs">10 m (Native Grid)</span>
                    <span className="text-[10px] opacity-75 font-normal">128 × 128 px raster</span>
                  </button>
                </div>

                <div className="space-y-1.5 font-mono text-[11px] text-slate-400">
                  <div className="flex justify-between"><span>Resampling:</span> <span className="text-slate-200">Nearest Neighbor</span></div>
                  <div className="flex justify-between"><span>Alignment:</span> <span className="text-slate-200">Affine Transform Match</span></div>
                  <div className="flex justify-between"><span>Ignore/Nodata:</span> <span className="text-slate-200">Class 255</span></div>
                </div>
              </div>
            </div>

            {/* 3. Export Artifacts (Section 18) */}
            <div className="card p-5 space-y-4 border border-white/[0.08]">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                <div className="flex items-center gap-2">
                  <Download className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                    Export Artifacts
                  </h3>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <a
                  href={groundTruthExportUrl(activeJobId, 'geojson')}
                  download={`ground_truth_${activeJobId}.geojson`}
                  className="btn-ghost p-2.5 h-auto text-left justify-start"
                >
                  <FileJson className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <div className="min-w-0">
                    <span className="font-semibold block truncate">GeoJSON</span>
                    <span className="text-[10px] text-slate-500">Vector Polygons</span>
                  </div>
                </a>

                <a
                  href={groundTruthExportUrl(activeJobId, 'geotiff')}
                  download={`ground_truth_${activeJobId}.tif`}
                  className="btn-ghost p-2.5 h-auto text-left justify-start"
                >
                  <Layers className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                  <div className="min-w-0">
                    <span className="font-semibold block truncate">GeoTIFF</span>
                    <span className="text-[10px] text-slate-500">GT Label Raster</span>
                  </div>
                </a>

                <a
                  href={groundTruthExportUrl(activeJobId, 'metadata')}
                  download={`ground_truth_metadata_${activeJobId}.json`}
                  className="btn-ghost p-2.5 h-auto text-left justify-start"
                >
                  <FileJson className="w-4 h-4 text-amber-400 flex-shrink-0" />
                  <div className="min-w-0">
                    <span className="font-semibold block truncate">Metadata</span>
                    <span className="text-[10px] text-slate-500">Provenance JSON</span>
                  </div>
                </a>

                <a
                  href={groundTruthExportUrl(activeJobId, 'csv')}
                  download={`ground_truth_stats_${activeJobId}.csv`}
                  className="btn-ghost p-2.5 h-auto text-left justify-start"
                >
                  <FileSpreadsheet className="w-4 h-4 text-purple-400 flex-shrink-0" />
                  <div className="min-w-0">
                    <span className="font-semibold block truncate">Class CSV</span>
                    <span className="text-[10px] text-slate-500">Area Statistics</span>
                  </div>
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* ──────────────────────────────────────────────────────────── */}
        {/* 5. Validation & Raster Statistics Report (Section 4 & 5)     */}
        {/* ──────────────────────────────────────────────────────────── */}
        {validationReport && (
          <div className={`card p-6 space-y-4 border ${
            validationReport.is_valid
              ? 'border-emerald-500/30 bg-emerald-950/10'
              : 'border-amber-500/30 bg-amber-950/10'
          }`}>
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div className="flex items-center gap-2.5">
                {validationReport.is_valid ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                )}
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Annotation Validation Report
                  </h3>
                  <span className="text-xs text-slate-400">
                    {validationReport.is_valid ? 'All geometry and semantic integrity checks passed' : 'Validation issues detected'}
                  </span>
                </div>
              </div>
              <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full border ${
                validationReport.is_valid
                  ? 'border-emerald-500/40 text-emerald-300 bg-emerald-500/20'
                  : 'border-amber-500/40 text-amber-300 bg-amber-500/20'
              }`}>
                {validationReport.is_valid ? 'VALIDATED' : 'ISSUES DETECTED'}
              </span>
            </div>

            {/* Errors */}
            {validationReport.errors?.length > 0 && (
              <div className="space-y-1.5 text-xs text-red-300">
                <span className="font-bold uppercase tracking-wider text-[11px] block">Errors:</span>
                <ul className="list-disc list-inside space-y-1">
                  {validationReport.errors.map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Warnings */}
            {validationReport.warnings?.length > 0 && (
              <div className="space-y-1.5 text-xs text-amber-300">
                <span className="font-bold uppercase tracking-wider text-[11px] block">Scientific Warnings:</span>
                <ul className="list-disc list-inside space-y-1">
                  {validationReport.warnings.map((warn, i) => (
                    <li key={i}>{warn}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Raster Area Statistics if available */}
        {gtData?.raster_stats?.class_area_statistics && (
          <div className="card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div>
                <h3 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
                  Rasterized Ground Truth Statistics
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Pixel coverage and physical area distribution across the common evaluation grid
                </p>
              </div>
              <span className="text-xs font-mono text-emerald-400">
                Coverage: {gtData.raster_stats.coverage_percentage ?? 100}%
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-xs">
              {Object.entries(gtData.raster_stats.class_area_statistics).map(([cName, stats]) => {
                const cls = GT_CLASSES.find(c => c.name === cName) || { color: '#94a3b8' };
                return (
                  <div key={cName} className="p-3 rounded-xl bg-slate-900/60 border border-white/[0.05] space-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cls.color }} />
                      <span className="font-bold text-slate-200 truncate">{cName}</span>
                    </div>
                    <div className="font-mono text-slate-400 text-[11px] space-y-0.5">
                      <div>Area: <strong className="text-slate-200">{stats.area_ha ?? '—'} ha</strong></div>
                      <div>Share: <strong className="text-cyan-300">{stats.percent ?? '—'}%</strong></div>
                      <div>Pixels: <span>{stats.pixel_count ?? '—'}</span></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
