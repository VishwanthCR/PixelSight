import React, { useState } from 'react';
import { Upload, Database, Sliders, Play, Building2, Sprout, Waves, FileCode } from 'lucide-react';

export default function UploadPanel({ sampleRegions, onRunAnalysis, loading }) {
  const [inputMode, setInputMode] = useState('region'); // 'region' or 'upload'
  const [selectedRegion, setSelectedRegion] = useState('region2'); // default Urban Chennai
  const [patchIndex, setPatchIndex] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [useCase, setUseCase] = useState('urban');
  const [nSamples, setNSamples] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    onRunAnalysis({
      useCase,
      nSamples,
      regionId: inputMode === 'region' ? selectedRegion : null,
      patchIndex: inputMode === 'region' ? patchIndex : 0,
      file: inputMode === 'upload' ? selectedFile : null,
    });
  };

  return (
    <div className="glass-card rounded-2xl p-5 mb-8">
      <form onSubmit={handleSubmit} className="space-y-6">
        
        {/* Section Title */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-blue-400" />
              Patch Selection & Analysis Lens Configuration
            </h2>
            <p className="text-xs text-slate-400">
              Select Sentinel-2 4-channel patch (R, G, B, NIR) and choose your task evaluation lens
            </p>
          </div>
          
          {/* Mode Switcher */}
          <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs font-medium">
            <button
              type="button"
              onClick={() => setInputMode('region')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                inputMode === 'region' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Region .npy Datasets
            </button>
            <button
              type="button"
              onClick={() => setInputMode('upload')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                inputMode === 'upload' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Upload Custom Patch
            </button>
          </div>
        </div>

        {/* 1. Input Source Selection */}
        {inputMode === 'region' ? (
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Select Provided Region Dataset (R, G, B, NIR)
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {['region1', 'region2', 'region3'].map((regId) => {
                const info = sampleRegions?.[regId] || {
                  name: regId === 'region1' ? 'Region 1 (Agriculture)' : regId === 'region2' ? 'Region 2 (Urban - Chennai)' : 'Region 3 (Coastal / Disaster)',
                  description: '4-channel Sentinel-2 dataset',
                  primary_lens: regId === 'region1' ? 'crop' : regId === 'region2' ? 'urban' : 'disaster',
                  loaded: false
                };
                const isSelected = selectedRegion === regId;

                return (
                  <div
                    key={regId}
                    onClick={() => {
                      setSelectedRegion(regId);
                      setUseCase(info.primary_lens || 'urban');
                    }}
                    className={`cursor-pointer rounded-xl p-3.5 border transition-all ${
                      isSelected
                        ? 'bg-blue-950/40 border-blue-500 shadow-lg shadow-blue-950/50'
                        : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4 text-blue-400" />
                        <span className="text-sm font-semibold text-white">{info.name}</span>
                      </div>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono ${
                        info.loaded ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                      }`}>
                        {info.loaded ? 'Ready' : '# TODO: Awaiting .npy'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 line-clamp-2">{info.description}</p>
                  </div>
                );
              })}
            </div>

            {/* Patch Index Selector */}
            <div className="mt-3 flex items-center gap-3 bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <label className="text-xs text-slate-300 whitespace-nowrap">
                Patch Index (0 - 1999):
              </label>
              <input
                type="number"
                min="0"
                max="1999"
                value={patchIndex}
                onChange={(e) => setPatchIndex(parseInt(e.target.value) || 0)}
                className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1 text-xs text-white font-mono w-28 focus:outline-none focus:border-blue-500"
              />
              <span className="text-xs text-slate-500">
                Extracts (64, 64, 4) target & (32, 32, 4) LR input patch
              </span>
            </div>
          </div>
        ) : (
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Upload Custom Sentinel-2 Patch (.npy, .tif, .png)
            </label>
            <div className="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-6 text-center bg-slate-900/40 transition-colors">
              <input
                type="file"
                accept=".npy,.tif,.tiff,.png,.jpg"
                onChange={(e) => setSelectedFile(e.target.files[0])}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-2">
                <Upload className="w-8 h-8 text-blue-400 animate-bounce" />
                <span className="text-sm font-medium text-slate-200">
                  {selectedFile ? selectedFile.name : 'Click to select or drag Sentinel-2 patch file'}
                </span>
                <span className="text-xs text-slate-400">
                  Supports 4-channel float32 array (.npy) or image files
                </span>
              </label>
            </div>
          </div>
        )}

        {/* 2. Use-Case Lens Selector */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
            Select Use-Case Evaluation Lens
          </label>
          <div className="grid grid-cols-3 gap-3">
            {[
              { id: 'urban', label: 'Urban Infrastructure', icon: Building2, desc: 'Emphasize Building Edges & SSIM' },
              { id: 'crop', label: 'Crop / Agriculture', icon: Sprout, desc: 'Emphasize NDVI Vegetation Health' },
              { id: 'disaster', label: 'Disaster / Coastal', icon: Waves, desc: 'Emphasize NDWI Water & Flood Risks' },
            ].map((lens) => {
              const Icon = lens.icon;
              const isActive = useCase === lens.id;
              return (
                <div
                  key={lens.id}
                  onClick={() => setUseCase(lens.id)}
                  className={`cursor-pointer rounded-xl p-3 border transition-all flex items-start gap-3 ${
                    isActive
                      ? 'bg-blue-600/20 border-blue-500 text-white shadow-md'
                      : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <Icon className={`w-5 h-5 mt-0.5 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                  <div>
                    <div className="text-xs font-bold text-white">{lens.label}</div>
                    <div className="text-[11px] text-slate-400">{lens.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 3. Controls & Run Button */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pt-2">
          {/* MC Dropout Iterations Slider */}
          <div className="flex items-center gap-3 bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800 w-full md:w-auto">
            <span className="text-xs text-slate-300 whitespace-nowrap">
              MC Dropout Passes: <strong className="text-blue-400 font-mono">{nSamples}</strong>
            </span>
            <input
              type="range"
              min="5"
              max="30"
              value={nSamples}
              onChange={(e) => setNSamples(parseInt(e.target.value))}
              className="w-32 accent-blue-500 cursor-pointer"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full md:w-auto px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-semibold text-sm rounded-xl shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Processing SR Analysis...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                Run SR & Analytics Pipeline
              </>
            )}
          </button>
        </div>

      </form>
    </div>
  );
}
