import React, { useState, useEffect } from 'react';
import {
  Sprout,
  Building2,
  Flame,
  Zap,
  MapPin,
  Upload,
  ArrowLeft,
  ChevronDown,
  ChevronUp,
  Play,
  ShieldCheck,
  AlertTriangle,
  Info,
  CheckCircle2,
  Sparkles,
  Layers,
  Cpu
} from 'lucide-react';
import SatelliteMap from '../components/SatelliteMap.jsx';
import SceneSelector from '../components/SceneSelector.jsx';
import {
  searchCopernicusScenes,
  estimateCopernicusAoi,
  startCopernicusJob,
  startCropProcessing,
  startUrbanProcessing,
  startDisasterProcessing,
  startProcessing,
  inspectImage
} from '../api/srmApi.js';

export default function ApplicationInputDashboard({
  application = 'crop',
  onBack,
  onJobStarted,
}) {
  const [inputMode, setInputMode] = useState('map'); // 'map' | 'upload'
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Map & Copernicus State - Natural default 1.28km x 1.28km AOI (128x128 pixels, 1 tile)
  const [selectedAoi, setSelectedAoi] = useState([80.264798, 13.076941, 80.276602, 13.088459]);
  const [startDate, setStartDate] = useState('2023-06-01');
  const [endDate, setEndDate] = useState('2023-06-30');
  const [maxCloudCover, setMaxCloudCover] = useState(25.0);
  const [scenes, setScenes] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedScene, setSelectedScene] = useState(null);

  const handleAoiChange = (newAoi) => {
    setSelectedAoi(newAoi);
    setScenes([]);
    setSelectedScene(null);
    setSelectedPreScene(null);
    setSelectedPostScene(null);
  };

  // Disaster specific scenes
  const [selectedPreScene, setSelectedPreScene] = useState(null);
  const [selectedPostScene, setSelectedPostScene] = useState(null);

  // Upload Mode State
  const [uploadFile, setUploadFile] = useState(null);
  const [disasterPreFile, setDisasterPreFile] = useState(null);
  const [disasterPostFile, setDisasterPostFile] = useState(null);
  const [uploadInspection, setUploadInspection] = useState(null);
  const [isInspecting, setIsInspecting] = useState(false);

  // Common Submission / Error State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  // Application-specific themes and metadata
  const APP_INFO = {
    crop: {
      name: 'Crop Monitoring',
      subtitle: 'Multispectral NDVI Vegetation Health & Photosynthetic Canopy Assessment',
      icon: Sprout,
      color: 'emerald',
      gradient: 'from-emerald-500/20 via-sky-500/10 to-transparent',
      borderColor: 'border-emerald-500/30',
      badge: 'B04 / B08 NDVI Analysis',
    },
    urban: {
      name: 'Urban Analysis',
      subtitle: 'High-Resolution 7-Class Land-Cover Segmentation & Built-up Footprints',
      icon: Building2,
      color: 'amber',
      gradient: 'from-amber-500/20 via-sky-500/10 to-transparent',
      borderColor: 'border-amber-500/30',
      badge: 'WorldCover 7-Class Map',
    },
    disaster: {
      name: 'Disaster Management',
      subtitle: 'Temporal Baseline vs Impacted Scene Spectral Change & Damage Evaluation',
      icon: Flame,
      color: 'rose',
      gradient: 'from-rose-500/20 via-amber-500/10 to-transparent',
      borderColor: 'border-rose-500/30',
      badge: 'Bitemporal Alignment',
    },
    research: {
      name: 'Research Engine',
      subtitle: 'Pure 100-Step LDSR-S2 Latent Diffusion Super-Resolution & Stochastic Uncertainty',
      icon: Zap,
      color: 'cyan',
      gradient: 'from-cyan-500/20 via-blue-500/10 to-transparent',
      borderColor: 'border-cyan-500/30',
      badge: '100-Step Diffusion Core',
    },
  };

  const currentApp = APP_INFO[application] || APP_INFO.crop;
  const Icon = currentApp.icon;

  // Search scenes handler
  const handleSearchScenes = async () => {
    if (!selectedAoi) return;
    setIsSearching(true);
    setError(null);
    try {
      const resp = await searchCopernicusScenes(selectedAoi, startDate, endDate, maxCloudCover, 20);
      setScenes(resp.scenes || []);
      if (resp.best_scene) {
        if (application === 'disaster') {
          if (!selectedPreScene) setSelectedPreScene(resp.best_scene);
        } else {
          setSelectedScene(resp.best_scene);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to search Copernicus Catalog.');
    } finally {
      setIsSearching(false);
    }
  };

  // Upload inspection handler
  const handleFileChange = async (file, type = 'single') => {
    setError(null);
    if (type === 'single') {
      setUploadFile(file);
      setIsInspecting(true);
      try {
        const insp = await inspectImage(file);
        setUploadInspection(insp);
      } catch (err) {
        setUploadInspection(null);
        setError(err.message);
      } finally {
        setIsInspecting(false);
      }
    } else if (type === 'disaster_pre') {
      setDisasterPreFile(file);
    } else if (type === 'disaster_post') {
      setDisasterPostFile(file);
    }
  };

  // Launch analysis
  const handleStartAnalysis = async () => {
    setError(null);
    setIsSubmitting(true);

    try {
      let createdJob;
      if (inputMode === 'map') {
        if (!selectedAoi) {
          throw new Error('Please select an Area of Interest (AOI) on the map.');
        }

        if (application === 'disaster') {
          if (!selectedPreScene || !selectedPostScene) {
            throw new Error('Disaster analysis requires both a Pre-Event Scene and a Post-Event Scene.');
          }
          createdJob = await startCopernicusJob('disaster', {
            aoi: selectedAoi,
            pre_scene_id: selectedPreScene.id,
            pre_date: selectedPreScene.datetime,
            post_scene_id: selectedPostScene.id,
            post_date: selectedPostScene.datetime,
          });
        } else {
          if (!selectedScene) {
            throw new Error('Please choose a candidate Sentinel-2 scene from the list.');
          }
          createdJob = await startCopernicusJob(application, {
            aoi: selectedAoi,
            scene_id: selectedScene.id,
            date: selectedScene.datetime,
          });
        }
      } else {
        // Upload Mode
        if (application === 'disaster') {
          if (!disasterPreFile || !disasterPostFile) {
            throw new Error('Please upload both pre-event and post-event Sentinel-2 GeoTIFF files.');
          }
          createdJob = await startDisasterProcessing(disasterPreFile, disasterPostFile);
        } else if (application === 'crop') {
          if (!uploadFile) throw new Error('Please upload a Sentinel-2 GeoTIFF file.');
          createdJob = await startCropProcessing(uploadFile);
        } else if (application === 'urban') {
          if (!uploadFile) throw new Error('Please upload a Sentinel-2 GeoTIFF file.');
          createdJob = await startUrbanProcessing(uploadFile);
        } else {
          if (!uploadFile) throw new Error('Please upload a Sentinel-2 GeoTIFF file.');
          createdJob = await startProcessing(uploadFile);
        }
      }

      if (onJobStarted) {
        onJobStarted(createdJob);
      }
    } catch (err) {
      setError(err.message || 'Failed to initiate processing job.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen pb-20 px-4 md:px-8 max-w-7xl mx-auto text-slate-200">
      {/* Top Header */}
      <header className="py-6 flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] mb-6">
        <div className="flex items-center gap-3.5">
          <button
            onClick={onBack}
            className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
            title="Return to Application Selection"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2.5">
              <div className={`p-2 rounded-xl bg-${currentApp.color}-500/10 text-${currentApp.color}-400 border ${currentApp.borderColor}`}>
                <Icon className="w-5 h-5" />
              </div>
              <h1 className="text-xl md:text-2xl font-black text-white tracking-tight">
                {currentApp.name} Dashboard
              </h1>
              <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase bg-${currentApp.color}-500/10 text-${currentApp.color}-300 border ${currentApp.borderColor}`}>
                {currentApp.badge}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">{currentApp.subtitle}</p>
          </div>
        </div>

        {/* Dual Input Mode Tabs */}
        <div className="flex items-center bg-slate-900/90 rounded-xl p-1 border border-slate-800 shadow-inner">
          <button
            onClick={() => setInputMode('map')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              inputMode === 'map'
                ? 'bg-sky-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MapPin className="w-3.5 h-3.5 text-sky-300" />
            Option A — Sentinel-2 Satellite Map
          </button>
          <button
            onClick={() => setInputMode('upload')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              inputMode === 'upload'
                ? 'bg-sky-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Upload className="w-3.5 h-3.5 text-sky-300" />
            Option B — Upload Data
          </button>
        </div>
      </header>

      {/* Error Alert */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-start gap-3 text-xs shadow-lg animate-fade-in">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
          <div className="flex-1">
            <div className="font-bold text-rose-200">Attention Required</div>
            <div className="mt-0.5 leading-relaxed">{error}</div>
          </div>
          <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200 text-base">✕</button>
        </div>
      )}

      {/* Main Workflow View */}
      {inputMode === 'map' ? (
        <div className="space-y-6">
          {/* Satellite Map */}
          <SatelliteMap
            selectedAoi={selectedAoi}
            onAoiChange={setSelectedAoi}
            startDate={startDate}
            onStartDateChange={setStartDate}
            endDate={endDate}
            onEndDateChange={setEndDate}
            maxCloudCover={maxCloudCover}
            onMaxCloudCoverChange={setMaxCloudCover}
            onSearchScenes={handleSearchScenes}
            isSearching={isSearching}
          />

          {/* Scene Selector */}
          <SceneSelector
            scenes={scenes}
            selectedScene={selectedScene}
            onSelectScene={setSelectedScene}
            onAutoSelectBest={() => {
              if (scenes.length > 0) {
                const best = [...scenes].sort((a, b) => a.cloud_cover - b.cloud_cover)[0];
                setSelectedScene(best);
              }
            }}
            isDisaster={application === 'disaster'}
            selectedPreScene={selectedPreScene}
            onSelectPreScene={setSelectedPreScene}
            selectedPostScene={selectedPostScene}
            onSelectPostScene={setSelectedPostScene}
          />

          {/* Acquisition & Processing Summary Confirmation Card */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl flex flex-wrap items-center justify-between gap-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Target Acquisition Summary
                </span>
                <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/20">
                  Ready for LDSR-S2
                </span>
              </div>
              <div className="text-xs text-slate-400">
                {application === 'disaster' ? (
                  <span>
                    Pre: <strong className="text-slate-200">{selectedPreScene ? selectedPreScene.datetime.split('T')[0] : 'None'}</strong> | Post: <strong className="text-slate-200">{selectedPostScene ? selectedPostScene.datetime.split('T')[0] : 'None'}</strong>
                  </span>
                ) : (
                  <span>
                    Scene: <strong className="text-slate-200">{selectedScene ? `${selectedScene.datetime.split('T')[0]} (${selectedScene.cloud_cover}% cloud)` : 'No scene selected yet'}</strong>
                  </span>
                )}
                {' · '}Bands: <strong className="text-slate-200">B02, B03, B04, B08 (10m)</strong>
              </div>
            </div>

            {/* Start Processing Action */}
            <button
              onClick={handleStartAnalysis}
              disabled={isSubmitting || (application === 'disaster' ? (!selectedPreScene || !selectedPostScene) : !selectedScene)}
              className={`flex items-center gap-2.5 px-6 py-3 rounded-xl font-bold text-xs uppercase tracking-wider transition-all shadow-xl ${
                isSubmitting || (application === 'disaster' ? (!selectedPreScene || !selectedPostScene) : !selectedScene)
                  ? 'bg-slate-800 text-slate-500 border border-slate-700/50 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-500 to-sky-600 hover:from-emerald-400 hover:to-sky-500 text-white border border-emerald-400/40 hover:shadow-emerald-500/20 active:scale-95'
              }`}
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Initiating Pipeline...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Start {currentApp.name} (100 Steps)
                </>
              )}
            </button>
          </div>
        </div>
      ) : (
        /* Upload Mode View */
        <div className="space-y-6">
          <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-6">
            <div>
              <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <Upload className="w-4 h-4 text-sky-400" />
                Upload Sentinel-2 Multispectral GeoTIFF
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Provide a 4-band Sentinel-2 L2A GeoTIFF containing B02, B03, B04, and B08. Standard RGB image formats (PNG/JPEG) are strictly rejected to ensure scientific fidelity.
              </p>
            </div>

            {application === 'disaster' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Pre-event Upload */}
                <div className="p-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 flex flex-col items-center justify-center text-center p-6 space-y-3">
                  <span className="text-xs font-bold text-sky-400 uppercase">Pre-Event Baseline GeoTIFF</span>
                  <input
                    type="file"
                    accept=".tif,.tiff"
                    onChange={(e) => handleFileChange(e.target.files[0], 'disaster_pre')}
                    className="text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-500 cursor-pointer"
                  />
                  {disasterPreFile && (
                    <span className="text-xs font-mono text-emerald-400">✓ {disasterPreFile.name}</span>
                  )}
                </div>

                {/* Post-event Upload */}
                <div className="p-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 flex flex-col items-center justify-center text-center p-6 space-y-3">
                  <span className="text-xs font-bold text-amber-400 uppercase">Post-Event Impacted GeoTIFF</span>
                  <input
                    type="file"
                    accept=".tif,.tiff"
                    onChange={(e) => handleFileChange(e.target.files[0], 'disaster_post')}
                    className="text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-amber-600 file:text-white hover:file:bg-amber-500 cursor-pointer"
                  />
                  {disasterPostFile && (
                    <span className="text-xs font-mono text-amber-400">✓ {disasterPostFile.name}</span>
                  )}
                </div>
              </div>
            ) : (
              /* Single file upload */
              <div className="p-8 rounded-xl border-2 border-dashed border-slate-700 hover:border-sky-500/60 bg-slate-950/40 transition-colors flex flex-col items-center justify-center text-center space-y-4">
                <div className="p-3 rounded-2xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  <Upload className="w-8 h-8" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-200">Drag and drop your Sentinel-2 GeoTIFF here</div>
                  <div className="text-xs text-slate-500 mt-0.5">Supports .tif and .tiff with B02, B03, B04, B08</div>
                </div>
                <input
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  onChange={(e) => handleFileChange(e.target.files[0], 'single')}
                  className="text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-bold file:bg-sky-600 file:text-white hover:file:bg-sky-500 cursor-pointer"
                />
                {isInspecting && (
                  <div className="text-xs text-sky-400 animate-pulse">Inspecting raster headers and band tags...</div>
                )}
                {uploadInspection && (
                  <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-mono">
                    ✓ Valid Sentinel-2 GeoTIFF: {uploadInspection.bands} bands ({uploadInspection.band_names.join(', ')}), CRS: {uploadInspection.crs}
                  </div>
                )}
              </div>
            )}

            {/* Launch Upload Pipeline */}
            <div className="flex justify-end pt-2">
              <button
                onClick={handleStartAnalysis}
                disabled={isSubmitting || (application === 'disaster' ? (!disasterPreFile || !disasterPostFile) : !uploadFile)}
                className={`flex items-center gap-2.5 px-6 py-3 rounded-xl font-bold text-xs uppercase tracking-wider transition-all shadow-xl ${
                  isSubmitting || (application === 'disaster' ? (!disasterPreFile || !disasterPostFile) : !uploadFile)
                    ? 'bg-slate-800 text-slate-500 border border-slate-700/50 cursor-not-allowed'
                    : 'bg-gradient-to-r from-emerald-500 to-sky-600 hover:from-emerald-400 hover:to-sky-500 text-white border border-emerald-400/40 hover:shadow-emerald-500/20 active:scale-95'
                }`}
              >
                <Play className="w-4 h-4 fill-white" />
                Start Analysis (100 Steps)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Technical Details Accordion (Rule 31) */}
      <div className="mt-8 border border-slate-800/80 rounded-2xl overflow-hidden bg-slate-950/40">
        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-sky-400" /> Technical Details & Execution Specs
          </span>
          {showTechnicalDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showTechnicalDetails && (
          <div className="p-5 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="space-y-1.5">
              <div className="font-bold text-slate-300">Spectral Architecture</div>
              <div className="text-slate-400">Inputs: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)</div>
              <div className="text-slate-400">Native Resolution: 10 m Ground Sampling Distance (GSD)</div>
              <div className="text-slate-400">Output Scale: 4× super-resolved representation (~2.5 m equivalent)</div>
            </div>

            <div className="space-y-1.5">
              <div className="font-bold text-slate-300">Inference Core</div>
              <div className="text-slate-400">Model: LDSR-S2 Latent Diffusion</div>
              <div className="text-slate-400">Diffusion Sampling Steps: Exactly 100 steps (uncompromised)</div>
              <div className="text-slate-400">Batch Size: 1 tile (tuned for RTX 3050 6GB VRAM safety)</div>
            </div>

            <div className="space-y-1.5">
              <div className="font-bold text-slate-300">Copernicus Provenance</div>
              <div className="text-slate-400">Provider: Copernicus Data Space Ecosystem (CDSE)</div>
              <div className="text-slate-400">Data Collection: Sentinel-2 L2A BOA Reflectance</div>
              <div className="text-slate-400">Geospatial CRS: Retained from EPSG:4326/UTM source</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
