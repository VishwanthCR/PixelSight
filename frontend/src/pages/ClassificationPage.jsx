import React, { useState, useEffect, useRef } from 'react';
import {
  Map, Upload, CheckCircle2, AlertTriangle, Download, RotateCcw,
  Sparkles, Layers, Info, ArrowLeft, Eye, ShieldAlert, Cpu
} from 'lucide-react';
import CompareSlider from '../components/CompareSlider.jsx';
import { fetchSegmentationStatus, startSegmentation, inspectImage } from '../api/srmApi.js';

const DEFAULT_CLASSES = [
  { id: 0, name: 'Tree', color: '#28b45a', desc: 'Trees & closed forest canopy' },
  { id: 1, name: 'Shrubland', color: '#78aa50', desc: 'Shrub and bush formations' },
  { id: 2, name: 'Grassland', color: '#aad264', desc: 'Natural & semi-natural herbaceous' },
  { id: 3, name: 'Cropland', color: '#dcbe46', desc: 'Cultivated agricultural fields' },
  { id: 4, name: 'Built-up', color: '#d25a37', desc: 'Impervious structures & building clusters' },
  { id: 5, name: 'Bare', color: '#96876e', desc: 'Bare soil, sand, and rock surfaces' },
  { id: 6, name: 'Water', color: '#327dd2', desc: 'Permanent and seasonal open water' },
  { id: 255, name: 'Ignore', color: '#1e293b', desc: 'No-data / unclassified background mask' },
];

export default function ClassificationPage({ onBack }) {
  const [modelStatus, setModelStatus] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorDetails, setErrorDetails] = useState(null);
  const [results, setResults] = useState(null);
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'side-by-side'
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchSegmentationStatus()
      .then(setModelStatus)
      .catch(() => {
        setModelStatus({
          available: false,
          reason: "Unable to reach segmentation status endpoint.",
          expected_resource: "checkpoints/segmentation/unet_worldcover_best.pth",
          how_to_fix: "Check that the PixelSight backend server is running and checkpoints directory exists.",
        });
      });
  }, []);

  const handleFileChange = async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorDetails(null);
    setResults(null);
    setSelectedFile(file);
    try {
      const insp = await inspectImage(file);
      setInspection(insp);
    } catch (err) {
      setInspection(null);
      setErrorDetails({
        title: 'Inspection failed',
        reason: err.message,
        expected_resource: 'Valid Sentinel-2 4-band GeoTIFF (B02, B03, B04, B08)',
        how_to_fix: 'Provide a GeoTIFF image with valid band tags matching Sentinel-2 specification.',
      });
    }
  };

  const handleRunClassification = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setErrorDetails(null);
    try {
      const res = await startSegmentation(selectedFile);
      setResults(res);
    } catch (err) {
      // Truthful error parsing
      setErrorDetails({
        title: 'Classification unavailable',
        reason: err.message || 'Model execution or resource access failed.',
        expected_resource: 'checkpoints/segmentation/unet_worldcover_best.pth',
        how_to_fix: 'Verify that the UNet WorldCover weights are present at checkpoints/segmentation/unet_worldcover_best.pth and GPU/CPU has sufficient memory.',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setInspection(null);
    setResults(null);
    setErrorDetails(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const urbanData = results?.results || {};
  const classesList = modelStatus?.classes || DEFAULT_CLASSES;

  return (
    <div className="min-h-screen pb-20 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-8">
        <div className="flex items-center gap-3.5">
          <button
            onClick={onBack}
            className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
            title="Back to Landing Page"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Map className="w-4 h-4" />
              </div>
              <h1 className="text-xl font-bold text-white tracking-tight">Land Cover Classification</h1>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300">
                ESA WorldCover 7-Class
              </span>
            </div>
            <div className="text-xs text-slate-400 mt-1 font-medium">
              Deep UNet pixel-level land-cover segmentation directly from Sentinel-2 multispectral GeoTIFF
            </div>
          </div>
        </div>

        {/* Model status chip */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                modelStatus?.available ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]' : 'bg-amber-400 animate-pulse'
              }`}
            />
            <span className="text-slate-300 font-mono text-[11px]">
              {modelStatus?.available ? `UNet Ready (${modelStatus.device?.toUpperCase() || 'CPU'})` : 'Checking Model...'}
            </span>
          </div>
          {(selectedFile || results) && (
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-slate-300 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          )}
        </div>
      </header>

      {/* Truthful Unavailable / Error Banner */}
      {errorDetails && (
        <div className="mb-8 rounded-2xl p-6 bg-red-950/40 border border-red-500/30 shadow-xl backdrop-blur-md space-y-4">
          <div className="flex items-center gap-3 text-red-400 font-bold text-base">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <span>{errorDetails.title}</span>
          </div>
          <div className="grid md:grid-cols-3 gap-4 text-xs">
            <div className="space-y-1 p-3.5 rounded-xl bg-red-900/20 border border-red-500/20">
              <span className="font-bold text-red-300 uppercase tracking-wider text-[10px]">Reason</span>
              <p className="text-slate-300 leading-relaxed">{errorDetails.reason}</p>
            </div>
            <div className="space-y-1 p-3.5 rounded-xl bg-red-900/20 border border-red-500/20">
              <span className="font-bold text-red-300 uppercase tracking-wider text-[10px]">Expected Resource</span>
              <p className="text-slate-300 font-mono text-[11px] break-all">{errorDetails.expected_resource}</p>
            </div>
            <div className="space-y-1 p-3.5 rounded-xl bg-red-900/20 border border-red-500/20">
              <span className="font-bold text-red-300 uppercase tracking-wider text-[10px]">How to Fix</span>
              <p className="text-slate-300 leading-relaxed">{errorDetails.how_to_fix}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      {!results ? (
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Upload Box */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative cursor-pointer rounded-3xl p-10 border-2 border-dashed border-cyan-500/30 hover:border-cyan-400 bg-slate-900/60 hover:bg-slate-900/90 transition-all duration-300 flex flex-col items-center justify-center text-center shadow-2xl"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-4 transition-transform group-hover:scale-110 shadow-lg shadow-cyan-500/20">
              <Upload className="w-8 h-8" />
            </div>
            <h2 className="text-lg font-bold text-white mb-1">
              {selectedFile ? selectedFile.name : 'Upload Sentinel-2 Scene for Classification'}
            </h2>
            <p className="text-xs text-slate-400 max-w-md leading-relaxed">
              Accepts 4-band GeoTIFF (B02 Blue, B03 Green, B04 Red, B08 NIR) up to 100 MB.
            </p>
            {selectedFile && (
              <div className="mt-4 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · {selectedFile.name}
              </div>
            )}
          </div>

          {/* Inspection Card if file selected */}
          {inspection && (
            <div className="card p-5 space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white uppercase tracking-wider">Raster Metadata</span>
                <span className={`px-2 py-0.5 rounded-full font-bold ${inspection.compatible ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-red-500/20 text-red-300 border border-red-500/30'}`}>
                  {inspection.compatible ? 'Compatible 4-Band' : 'Incompatible'}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                  <span className="text-[10px] uppercase text-slate-500 block">Dimensions</span>
                  <span className="font-mono font-bold text-slate-200">{inspection.width} × {inspection.height}</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                  <span className="text-[10px] uppercase text-slate-500 block">Bands</span>
                  <span className="font-mono font-bold text-cyan-300">{inspection.band_count} Bands</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                  <span className="text-[10px] uppercase text-slate-500 block">CRS</span>
                  <span className="font-mono font-bold text-slate-200">{inspection.crs || 'EPSG:32643'}</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/80 border border-white/[0.04]">
                  <span className="text-[10px] uppercase text-slate-500 block">Resolution</span>
                  <span className="font-mono font-bold text-slate-200">{inspection.resolution_m ? `${inspection.resolution_m} m` : '10 m'}</span>
                </div>
              </div>

              <button
                disabled={!inspection.compatible || isProcessing}
                onClick={handleRunClassification}
                className="w-full py-3.5 px-4 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg shadow-cyan-600/20 active:scale-95"
              >
                {isProcessing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Executing UNet Segmentation...</span>
                  </>
                ) : (
                  <>
                    <Layers className="w-4 h-4" />
                    <span>Run Land Cover Classification</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Class Legend Guide Card */}
          <div className="card p-5 space-y-3">
            <span className="text-xs uppercase font-bold tracking-wider text-slate-400 block">
              Supported ESA WorldCover Land-Cover Classes
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {classesList.map(c => (
                <div key={c.id} className="flex items-center gap-2.5 p-2 rounded-xl bg-slate-900/50 border border-white/[0.04] text-xs">
                  <span className="w-3.5 h-3.5 rounded-md flex-shrink-0 shadow-sm" style={{ backgroundColor: c.color }} />
                  <div className="min-w-0">
                    <span className="font-semibold text-slate-200 block truncate">{c.name}</span>
                    <span className="text-[10px] text-slate-500 font-mono">ID {c.id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Results View */
        <div className="space-y-8 anim-fade-in">
          {/* Main Visualizer Card */}
          <div className="card p-6 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Classified Surface Visualization</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setViewMode('slider')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    viewMode === 'slider' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Interactive Slider
                </button>
                <button
                  onClick={() => setViewMode('side-by-side')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    viewMode === 'side-by-side' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'bg-white/[0.04] text-slate-400'
                  }`}
                >
                  Side-by-Side
                </button>
              </div>
            </div>

            {viewMode === 'slider' ? (
              <div className="h-[460px] rounded-2xl overflow-hidden border border-white/[0.08] bg-black/40">
                <CompareSlider
                  leftSrc={results.preview_url}
                  rightSrc={results.preview_url}
                  leftLabel="Classification Map"
                  rightLabel="Land-Cover Raster"
                />
              </div>
            ) : (
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Predicted Land Cover Map</div>
                  <div className="rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 p-2 flex items-center justify-center">
                    <img src={results.preview_url} alt="Planning map" className="max-h-[380px] object-contain rounded-lg" />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Classification Class Map</div>
                  <div className="rounded-xl overflow-hidden border border-white/[0.08] bg-black/40 p-2 flex items-center justify-center">
                    <img src={results.preview_url} alt="Class map" className="max-h-[380px] object-contain rounded-lg" />
                  </div>
                </div>
              </div>
            )}

            {/* Downloads */}
            <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-white/[0.06]">
              <div className="text-xs text-slate-400">
                Digital GeoTIFF contains raster values [0..6, 255] aligned with Sentinel-2 coordinate reference system.
              </div>
              <div className="flex items-center gap-2">
                <a href={results.preview_url} download="classification_preview.png" className="btn-ghost text-xs">
                  <Download className="w-3.5 h-3.5" /> PNG Preview
                </a>
                <a href={results.raster_url} download="classification_classes.tif" className="btn-ghost text-xs">
                  <Download className="w-3.5 h-3.5" /> GeoTIFF Raster
                </a>
              </div>
            </div>
          </div>

          {/* Class Legend & Statistics Grid */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* 7-Class Interactive Legend */}
            <div className="md:col-span-2 card p-5 space-y-4">
              <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold block">
                Class-wise Object & Area Breakdown
              </span>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wider text-slate-500 border-b border-white/[0.08]">
                      <th className="py-2.5 pr-4">Class</th>
                      <th className="py-2.5 pr-4">Connected Objects</th>
                      <th className="py-2.5 pr-4">Pixel Count</th>
                      <th className="py-2.5">Area %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(urbanData.object_counts || {}).map(([name, item]) => {
                      const color = DEFAULT_CLASSES.find(c => c.name.toLowerCase() === name.toLowerCase())?.color || '#38bdf8';
                      return (
                        <tr key={name} className="border-b border-white/[0.04] text-slate-300">
                          <td className="py-2.5 pr-4 flex items-center gap-2.5">
                            <span className="w-3 h-3 rounded-md flex-shrink-0" style={{ backgroundColor: color }} />
                            <span className="font-medium text-white">{item.label || name}</span>
                          </td>
                          <td className="py-2.5 pr-4 font-mono text-cyan-300">{item.object_count ?? '—'}</td>
                          <td className="py-2.5 pr-4 font-mono text-slate-400">{item.pixel_count ?? '—'}</td>
                          <td className="py-2.5 font-mono text-slate-200">
                            {item.area_percent != null ? `${item.area_percent.toFixed(1)}%` : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Quick Metrics */}
            <div className="space-y-4">
              <div className="card p-5 space-y-3">
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold block">Key Regions</span>
                <div className="space-y-2.5">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Tree Clusters</span>
                    <span className="font-mono font-bold text-emerald-400">{urbanData.tree_clusters ?? 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Built-up Footprints</span>
                    <span className="font-mono font-bold text-amber-400">{urbanData.built_clusters ?? 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Total Valid Pixels</span>
                    <span className="font-mono font-bold text-cyan-300">{urbanData.total_pixels?.toLocaleString() ?? '—'}</span>
                  </div>
                </div>
              </div>

              {/* Scientific Notice */}
              <div className="card p-5 bg-amber-500/[0.04] border border-amber-500/20 text-xs text-amber-300/90 space-y-2">
                <div className="flex items-center gap-2 font-semibold text-amber-300">
                  <ShieldAlert className="w-4 h-4" />
                  <span>Scientific Proxy Disclaimer</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Classification is performed using an ESA WorldCover 10 m proxy model. Connected regions are statistical land-cover clusters, NOT official cadastral building footprints or land registry boundaries.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
