import React, { useState, useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, Rectangle, Marker, Popup, GeoJSON, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import {
  Layers,
  Crosshair,
  Maximize2,
  Calendar,
  Cloud,
  Search,
  CheckCircle2,
  RotateCcw,
  Sparkles,
  Info,
  AlertCircle,
  MapPin,
  Move,
  Sliders,
  Compass,
  ArrowRight,
  Loader2,
  X,
  ShieldCheck,
  Flag
} from 'lucide-react';
import { estimateCopernicusAoi, geocodePlace } from '../api/srmApi';
import indiaBoundaryData from '../data/indiaBoundary.json';

// Sleek draggable marker for AOI center handle
const aoiCenterIcon = L.divIcon({
  className: 'custom-aoi-center-pin',
  html: `
    <div style="
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: #0284c7;
      border: 2px solid #ffffff;
      border-radius: 50%;
      box-shadow: 0 4px 14px rgba(0,0,0,0.6);
      cursor: grab;
    ">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="22" y1="12" x2="18" y2="12"></line>
        <line x1="6" y1="12" x2="2" y2="12"></line>
        <line x1="12" y1="6" x2="12" y2="2"></line>
        <line x1="12" y1="22" x2="12" y2="18"></line>
      </svg>
    </div>
  `,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

// Sleek pin for place search results
const searchPinIcon = L.divIcon({
  className: 'custom-search-pin',
  html: `
    <div style="
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      background: #e11d48;
      border: 2px solid #ffffff;
      border-radius: 50%;
      box-shadow: 0 4px 12px rgba(225,29,72,0.5);
    ">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path>
        <circle cx="12" cy="10" r="3"></circle>
      </svg>
    </div>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

/**
 * Computes a bounding box [minLon, minLat, maxLon, maxLat] centered at (lat, lon)
 * for given dimensions in kilometers.
 */
export function computeBboxFromCenter(lat, lon, widthKm = 1.28, heightKm = 1.28) {
  const latRad = (lat * Math.PI) / 180;
  const deltaLat = heightKm / 111.132;
  const cosLat = Math.max(0.01, Math.cos(latRad));
  const deltaLon = widthKm / (111.320 * cosLat);

  const minLat = +(lat - deltaLat / 2).toFixed(6);
  const maxLat = +(lat + deltaLat / 2).toFixed(6);
  const minLon = +(lon - deltaLon / 2).toFixed(6);
  const maxLon = +(lon + deltaLon / 2).toFixed(6);

  return [minLon, minLat, maxLon, maxLat];
}

/**
 * Map fly-to controller
 */
function MapFlyController({ flyTarget }) {
  const map = useMap();
  useEffect(() => {
    if (!flyTarget) return;
    if (flyTarget.bounds) {
      map.fitBounds(flyTarget.bounds, { padding: [40, 40], maxZoom: 15 });
    } else if (flyTarget.center) {
      map.flyTo(flyTarget.center, flyTarget.zoom || 13, { duration: 1.2 });
    }
  }, [flyTarget, map]);
  return null;
}

/**
 * Handles map click, cursor movement, and freehand rectangle drawing
 */
function MapInteractionHandler({
  onCursorMove,
  onMapCenterChange,
  isDrawing,
  onDrawComplete,
  isClickToMoveActive,
  onClickToMove,
}) {
  const [startPoint, setStartPoint] = useState(null);
  const [currentPoint, setCurrentPoint] = useState(null);

  useMapEvents({
    mousemove(e) {
      onCursorMove([e.latlng.lat, e.latlng.lng]);
      if (isDrawing && startPoint) {
        setCurrentPoint([e.latlng.lat, e.latlng.lng]);
      }
    },
    moveend(e) {
      const c = e.target.getCenter();
      onMapCenterChange([c.lat, c.lng]);
    },
    click(e) {
      if (isClickToMoveActive) {
        onClickToMove(e.latlng.lat, e.latlng.lng);
        return;
      }
      if (!isDrawing) return;

      if (!startPoint) {
        setStartPoint([e.latlng.lat, e.latlng.lng]);
        setCurrentPoint([e.latlng.lat, e.latlng.lng]);
      } else {
        const p1 = startPoint;
        const p2 = [e.latlng.lat, e.latlng.lng];
        const minLat = Math.min(p1[0], p2[0]);
        const maxLat = Math.max(p1[0], p2[0]);
        const minLon = Math.min(p1[1], p2[1]);
        const maxLon = Math.max(p1[1], p2[1]);
        setStartPoint(null);
        setCurrentPoint(null);
        onDrawComplete([minLon, minLat, maxLon, maxLat]);
      }
    }
  });

  if (isDrawing && startPoint && currentPoint) {
    const bounds = [
      [Math.min(startPoint[0], currentPoint[0]), Math.min(startPoint[1], currentPoint[1])],
      [Math.max(startPoint[0], currentPoint[0]), Math.max(startPoint[1], currentPoint[1])]
    ];
    return (
      <Rectangle
        bounds={bounds}
        pathOptions={{ color: '#38bdf8', weight: 2, dashArray: '4, 4', fillOpacity: 0.2 }}
      />
    );
  }
  return null;
}

export default function SatelliteMap({
  selectedAoi,
  onAoiChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  maxCloudCover,
  onMaxCloudCoverChange,
  onSearchScenes,
  isSearching,
}) {
  const [basemap, setBasemap] = useState('satellite'); // 'satellite' | 'street'
  const [cursorPos, setCursorPos] = useState([13.0827, 80.2707]);
  const [mapCenter, setMapCenter] = useState([22.5, 79.5]); // Initial India Center
  const [flyTarget, setFlyTarget] = useState(null);

  // Search Places state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearchingPlace, setIsSearchingPlace] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [showResultsDropdown, setShowResultsDropdown] = useState(false);
  const [activePlaceMarker, setActivePlaceMarker] = useState(null);

  // AOI interaction modes
  const [isDrawing, setIsDrawing] = useState(false);
  const [isClickToMove, setIsClickToMove] = useState(false);
  const [activePreset, setActivePreset] = useState('default');
  const [customDimensions, setCustomDimensions] = useState({ widthKm: 1.28, heightKm: 1.28 });

  // Backend validation & estimation state
  const [estimation, setEstimation] = useState(null);
  const [estError, setEstError] = useState(null);

  // Quick Indian City Shortcuts for fast navigation
  const quickIndianCities = [
    { name: 'Chennai', lat: 13.0827, lon: 80.2707 },
    { name: 'Madurai', lat: 9.9252, lon: 78.1198 },
    { name: 'Coimbatore', lat: 11.0168, lon: 76.9558 },
    { name: 'Bengaluru', lat: 12.9716, lon: 77.5946 },
    { name: 'Hyderabad', lat: 17.3850, lon: 78.4867 },
    { name: 'Mumbai', lat: 19.0760, lon: 72.8777 },
    { name: 'Delhi', lat: 28.6139, lon: 77.2090 },
    { name: 'Kolkata', lat: 22.5726, lon: 88.3639 },
    { name: 'Kerala', lat: 10.8505, lon: 76.2711 },
  ];

  // AOI size presets (relative to center)
  const aoiPresets = [
    { id: 'small', label: 'Small (0.64 km)', size: 0.64, desc: '0.64 km × 0.64 km (~64² px, 1 tile)' },
    { id: 'default', label: 'Default (1.28 km)', size: 1.28, desc: '1.28 km × 1.28 km (128² px, 1 tile)' },
    { id: 'medium', label: 'Medium (2 km)', size: 2.0, desc: '2.0 km × 2.0 km (~200² px, 4 tiles)' },
    { id: 'large', label: 'Large (3 km)', size: 3.0, desc: '3.0 km × 3.0 km (~300² px, 9 tiles)' },
  ];

  // Derive center of current AOI or fallback to map center
  const aoiCenter = useMemo(() => {
    if (selectedAoi && selectedAoi.length === 4) {
      return [
        (selectedAoi[1] + selectedAoi[3]) / 2,
        (selectedAoi[0] + selectedAoi[2]) / 2,
      ];
    }
    return [13.0827, 80.2707];
  }, [selectedAoi]);

  // Leaflet rectangle bounds for selected AOI
  const aoiBounds = useMemo(() => {
    if (!selectedAoi || selectedAoi.length !== 4) return null;
    return [
      [selectedAoi[1], selectedAoi[0]],
      [selectedAoi[3], selectedAoi[2]],
    ];
  }, [selectedAoi]);

  // Ensure an initial natural 1.28 km default AOI exists inside India if none provided
  useEffect(() => {
    if (!selectedAoi) {
      const defaultBbox = computeBboxFromCenter(13.0827, 80.2707, 1.28, 1.28);
      onAoiChange(defaultBbox);
    }
  }, [selectedAoi, onAoiChange]);

  // Request backend estimation whenever selectedAoi changes
  useEffect(() => {
    if (!selectedAoi || selectedAoi.length !== 4) {
      setEstimation(null);
      setEstError(null);
      return;
    }
    let isMounted = true;
    estimateCopernicusAoi(selectedAoi)
      .then((res) => {
        if (isMounted) {
          setEstimation(res);
          setEstError(null);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setEstError(err.message || 'AOI validation error');
        }
      });
    return () => { isMounted = false; };
  }, [selectedAoi]);

  // Perform place search via backend geocoding endpoint (scoped to India)
  const handlePlaceSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim() || searchQuery.trim().length < 2) return;

    setIsSearchingPlace(true);
    setSearchError(null);
    setShowResultsDropdown(true);

    try {
      const resp = await geocodePlace(searchQuery.trim());
      setSearchResults(resp.results || []);
      if (!resp.results || resp.results.length === 0) {
        setSearchError(`No matching Indian locations found for "${searchQuery}".`);
      }
    } catch (err) {
      setSearchError('Geocoding service unavailable. Please check spelling or internet connection.');
    } finally {
      setIsSearchingPlace(false);
    }
  };

  // Select a place from search results
  const handleSelectPlace = (place) => {
    setShowResultsDropdown(false);
    setSearchQuery(place.display_name);

    if (place.inside_india === false) {
      setSearchError('PixelSight currently supports processing only within India.');
      return;
    }

    const targetCenter = [place.lat, place.lon];
    setActivePlaceMarker({
      lat: place.lat,
      lon: place.lon,
      title: place.display_name,
    });

    // Move map to the place
    setFlyTarget({ center: targetCenter, zoom: 13, timestamp: Date.now() });

    // Place the 1.28 km default AOI at the searched Indian place
    const widthKm = customDimensions.widthKm || 1.28;
    const heightKm = customDimensions.heightKm || 1.28;
    const newBbox = computeBboxFromCenter(place.lat, place.lon, widthKm, heightKm);
    onAoiChange(newBbox);
  };

  // Select a quick Indian city shortcut
  const handleQuickCity = (city) => {
    const targetCenter = [city.lat, city.lon];
    setSearchQuery(city.name);
    setActivePlaceMarker({
      lat: city.lat,
      lon: city.lon,
      title: city.name,
    });

    setFlyTarget({ center: targetCenter, zoom: 13, timestamp: Date.now() });

    const widthKm = customDimensions.widthKm || 1.28;
    const heightKm = customDimensions.heightKm || 1.28;
    const newBbox = computeBboxFromCenter(city.lat, city.lon, widthKm, heightKm);
    onAoiChange(newBbox);
  };

  // Apply an AOI size preset around the current AOI center
  const handleApplyPreset = (preset) => {
    setActivePreset(preset.id);
    setCustomDimensions({ widthKm: preset.size, heightKm: preset.size });
    const center = aoiCenter;
    const newBbox = computeBboxFromCenter(center[0], center[1], preset.size, preset.size);
    onAoiChange(newBbox);
  };

  // Handle center marker drag to reposition AOI
  const handleMarkerDragEnd = (e) => {
    const marker = e.target;
    const position = marker.getLatLng();
    const widthKm = estimation?.dimensions_km?.width_km || customDimensions.widthKm || 1.28;
    const heightKm = estimation?.dimensions_km?.height_km || customDimensions.heightKm || 1.28;
    const newBbox = computeBboxFromCenter(position.lat, position.lng, widthKm, heightKm);
    onAoiChange(newBbox);
  };

  // Click on map to reposition AOI center
  const handleClickToMove = (lat, lon) => {
    setIsClickToMove(false);
    const widthKm = estimation?.dimensions_km?.width_km || customDimensions.widthKm || 1.28;
    const heightKm = estimation?.dimensions_km?.height_km || customDimensions.heightKm || 1.28;
    const newBbox = computeBboxFromCenter(lat, lon, widthKm, heightKm);
    onAoiChange(newBbox);
  };

  // Center AOI on current map viewport center
  const handleCenterOnViewport = () => {
    const widthKm = estimation?.dimensions_km?.width_km || customDimensions.widthKm || 1.28;
    const heightKm = estimation?.dimensions_km?.height_km || customDimensions.heightKm || 1.28;
    const newBbox = computeBboxFromCenter(mapCenter[0], mapCenter[1], widthKm, heightKm);
    onAoiChange(newBbox);
  };

  // Fit map viewport to the AOI rectangle
  const handleFitAoi = () => {
    if (aoiBounds) {
      setFlyTarget({ bounds: aoiBounds, timestamp: Date.now() });
    }
  };

  // Complete drawing freehand rectangle
  const handleDrawComplete = (bbox) => {
    setIsDrawing(false);
    setActivePreset('custom');
    onAoiChange(bbox);
  };

  // Clear AOI
  const handleClearAoi = () => {
    setIsDrawing(false);
    setIsClickToMove(false);
    onAoiChange(null);
  };

  // Reset to full India view
  const handleResetToIndia = () => {
    setFlyTarget({ center: [22.5, 79.5], zoom: 5, timestamp: Date.now() });
  };

  // Validation checks for the checklist
  const isInsideIndia = estimation?.inside_india ?? !estError;
  const isValidAoi = !!selectedAoi && !estError;
  const isAreaWithinLimit = estimation ? estimation.area_sqkm <= 25.0 : true;

  return (
    <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
      {/* 1. TOP HEADER & INDIA SCOPE NOTICE */}
      <div className="p-3.5 bg-slate-900/95 border-b border-slate-800 space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Flag className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-100">
                  PixelSight India Satellite Explorer
                </span>
                <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  INDIA ONLY
                </span>
              </div>
              <div className="text-[11px] text-slate-400">
                Satellite super-resolution &amp; analysis is scoped to the sovereign geographic extent of India.
              </div>
            </div>
          </div>

          {/* Quick India View & Basemap Switcher */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleResetToIndia}
              className="px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/60 text-xs font-medium transition-colors"
              title="Reset viewport to full India view"
            >
              India Extent
            </button>

            <div className="flex items-center bg-slate-800/90 rounded-lg p-0.5 border border-slate-700/60 text-xs">
              <button
                onClick={() => setBasemap('satellite')}
                className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                  basemap === 'satellite'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Satellite
              </button>
              <button
                onClick={() => setBasemap('street')}
                className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                  basemap === 'street'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Carto
              </button>
            </div>
          </div>
        </div>

        {/* Real Indian Place Search with Autocomplete */}
        <div className="relative z-[1100]">
          <form onSubmit={handlePlaceSearch} className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowResultsDropdown(false);
                }}
                onFocus={() => {
                  if (searchResults.length > 0) setShowResultsDropdown(true);
                }}
                placeholder="Search Indian locations (e.g. Chennai, Madurai, Coimbatore, Bengaluru, Mumbai, Delhi, Kerala...)"
                className="w-full pl-9 pr-8 py-2 bg-slate-950 border border-slate-700/80 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 transition-colors shadow-inner"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                    setShowResultsDropdown(false);
                  }}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <button
              type="submit"
              disabled={isSearchingPlace || !searchQuery.trim()}
              className="px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-xs font-semibold rounded-xl border border-sky-500/80 shadow-md transition-all flex items-center gap-1.5 whitespace-nowrap"
            >
              {isSearchingPlace ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Search className="w-3.5 h-3.5" />
              )}
              <span>Locate Place</span>
            </button>
          </form>

          {/* Autocomplete Dropdown */}
          {showResultsDropdown && (
            <div className="absolute left-0 right-0 top-full mt-1.5 bg-slate-950 border border-slate-700/90 rounded-xl shadow-2xl max-h-56 overflow-y-auto z-[1200] divide-y divide-slate-800">
              {searchError ? (
                <div className="p-3 text-xs text-rose-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{searchError}</span>
                </div>
              ) : (
                searchResults.map((res) => (
                  <button
                    key={res.place_id}
                    onClick={() => handleSelectPlace(res)}
                    className="w-full text-left p-2.5 hover:bg-slate-800/80 text-xs text-slate-200 transition-colors flex items-start justify-between gap-2"
                  >
                    <div className="flex items-start gap-2">
                      <MapPin className="w-3.5 h-3.5 text-sky-400 mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-semibold text-slate-100">{res.display_name}</div>
                        <div className="text-[10px] text-slate-400">
                          Lat: {res.lat.toFixed(4)}°, Lon: {res.lon.toFixed(4)}°
                        </div>
                      </div>
                    </div>
                    {res.inside_india === false && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30 whitespace-nowrap">
                        Outside India
                      </span>
                    )}
                  </button>
                ))
              )}
            </div>
          )}
        </div>

        {/* Quick Indian City Shortcuts */}
        <div className="flex items-center gap-1.5 overflow-x-auto text-[11px] pt-0.5 pb-1">
          <span className="text-slate-400 font-medium whitespace-nowrap flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-amber-400" /> Indian Hubs:
          </span>
          {quickIndianCities.map((c) => (
            <button
              key={c.name}
              onClick={() => handleQuickCity(c)}
              className="px-2.5 py-0.5 rounded-full bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/60 whitespace-nowrap transition-colors"
            >
              {c.name}
            </button>
          ))}
        </div>
      </div>

      {/* 2. AOI SIZE PRESETS & ACTION TOOLBAR */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 px-3.5 py-2.5 bg-slate-950/70 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-slate-400 font-medium mr-1 flex items-center gap-1">
            <Maximize2 className="w-3 h-3 text-sky-400" /> AOI Size:
          </span>
          {aoiPresets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleApplyPreset(preset)}
              title={preset.desc}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all border ${
                activePreset === preset.id
                  ? 'bg-sky-500/20 text-sky-300 border-sky-500/60 font-semibold shadow-sm'
                  : 'bg-slate-800/60 hover:bg-slate-700 text-slate-300 border-slate-700/50'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Action Buttons: Reposition, Draw, Center, Clear */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => {
              setIsClickToMove(!isClickToMove);
              setIsDrawing(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
              isClickToMove
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/60 animate-pulse'
                : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border-slate-700/60'
            }`}
            title="Click anywhere inside India to reposition AOI center"
          >
            <Move className="w-3 h-3 text-amber-400" />
            <span>{isClickToMove ? 'Click map to place AOI' : 'Click to Move'}</span>
          </button>

          <button
            onClick={handleCenterOnViewport}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 text-xs font-medium transition-colors"
            title="Snap AOI to current map center"
          >
            <Crosshair className="w-3 h-3 text-sky-400" />
            <span>Center on Viewport</span>
          </button>

          <button
            onClick={() => {
              setIsDrawing(!isDrawing);
              setIsClickToMove(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
              isDrawing
                ? 'bg-sky-500/20 text-sky-300 border-sky-500/60 animate-pulse'
                : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border-slate-700/60'
            }`}
            title="Draw custom rectangle by clicking two corners"
          >
            <Sliders className="w-3 h-3 text-sky-400" />
            <span>{isDrawing ? 'Click opposite corners' : 'Draw Custom'}</span>
          </button>

          {selectedAoi && (
            <button
              onClick={handleClearAoi}
              className="p-1 text-slate-400 hover:text-slate-200 bg-slate-800/80 hover:bg-slate-700 rounded-lg border border-slate-700/60 transition-colors"
              title="Reset AOI"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* 3. INTERACTIVE MAP VIEWPORT (WITH INDIA BOUNDARY) */}
      <div className="relative h-[450px] w-full bg-slate-950">
        <MapContainer
          center={[22.5, 79.5]}
          zoom={5}
          style={{ height: '100%', width: '100%' }}
          scrollWheelZoom={true}
        >
          {basemap === 'satellite' ? (
            <TileLayer
              attribution='&copy; <a href="https://www.esri.com/">Esri</a>, Copernicus Data Space'
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              maxZoom={19}
            />
          ) : (
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              maxZoom={19}
            />
          )}

          {/* India Official Sovereign Boundary Visual Layer */}
          {indiaBoundaryData && (
            <GeoJSON
              data={indiaBoundaryData}
              style={{
                color: '#38bdf8',
                weight: 2,
                fillColor: '#0ea5e9',
                fillOpacity: 0.04,
                dashArray: '4, 6',
              }}
            />
          )}

          {/* Imperative fly controller */}
          <MapFlyController flyTarget={flyTarget} />

          {/* Mouse & pan interaction handler */}
          <MapInteractionHandler
            onCursorMove={setCursorPos}
            onMapCenterChange={setMapCenter}
            isDrawing={isDrawing}
            onDrawComplete={handleDrawComplete}
            isClickToMoveActive={isClickToMove}
            onClickToMove={handleClickToMove}
          />

          {/* Processing AOI Rectangle */}
          {aoiBounds && (
            <Rectangle
              bounds={aoiBounds}
              pathOptions={{
                color: estError ? '#f43f5e' : '#0284c7',
                weight: 2.5,
                fillColor: estError ? '#fda4af' : '#38bdf8',
                fillOpacity: 0.22,
                dashArray: isDrawing ? '4, 4' : null,
              }}
            />
          )}

          {/* Draggable Center Pin for Repositioning AOI */}
          {selectedAoi && aoiBounds && (
            <Marker
              position={aoiCenter}
              icon={aoiCenterIcon}
              draggable={true}
              eventHandlers={{
                dragend: handleMarkerDragEnd,
              }}
            >
              <Popup className="text-xs">
                <div className="font-semibold text-slate-800">PixelSight AOI Center</div>
                <div className="text-[11px] text-slate-600">Drag to reposition processing area within India</div>
              </Popup>
            </Marker>
          )}

          {/* Active Place Marker if searched */}
          {activePlaceMarker && (
            <Marker
              position={[activePlaceMarker.lat, activePlaceMarker.lon]}
              icon={searchPinIcon}
            >
              <Popup>
                <div className="font-semibold text-xs">{activePlaceMarker.title}</div>
                <div className="text-[10px] text-slate-500">Selected Location (India)</div>
              </Popup>
            </Marker>
          )}
        </MapContainer>

        {/* Status instruction banner when click-to-move or drawing */}
        {isClickToMove && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] px-4 py-1.5 bg-amber-500/95 backdrop-blur-md text-black font-semibold text-xs rounded-full shadow-lg border border-amber-300 flex items-center gap-1.5 animate-bounce">
            <Move className="w-3.5 h-3.5" />
            <span>Click anywhere inside India to place AOI center</span>
          </div>
        )}

        {isDrawing && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] px-4 py-1.5 bg-sky-500/95 backdrop-blur-md text-white font-semibold text-xs rounded-full shadow-lg border border-sky-300 flex items-center gap-1.5 animate-bounce">
            <Crosshair className="w-3.5 h-3.5" />
            <span>Click first corner, then opposite corner inside India</span>
          </div>
        )}

        {/* Floating Quick Action: Fit to AOI */}
        {selectedAoi && (
          <button
            onClick={handleFitAoi}
            className="absolute top-3 right-3 z-[1000] p-2 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/80 text-slate-300 hover:text-white hover:bg-slate-800 shadow-lg transition-colors"
            title="Zoom map to fit AOI"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        )}

        {/* Floating Bottom Left: Coordinates */}
        <div className="absolute bottom-3 left-3 z-[1000] px-2.5 py-1.5 rounded-lg bg-slate-900/85 backdrop-blur-md border border-slate-700/70 text-[11px] font-mono text-slate-300 pointer-events-none shadow-md flex items-center gap-3">
          <span>Lat: {cursorPos[0].toFixed(4)}°</span>
          <span className="text-slate-600">|</span>
          <span>Lon: {cursorPos[1].toFixed(4)}°</span>
        </div>

        {/* Floating Bottom Right: India Boundary & Processing Tag */}
        <div className="absolute bottom-3 right-3 z-[1000] px-3 py-1.5 rounded-lg bg-slate-900/85 backdrop-blur-md border border-slate-700/70 text-[11px] text-sky-400 font-semibold shadow-md flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${estError ? 'bg-rose-500' : 'bg-emerald-400 animate-ping'}`} />
          <span>{estError ? 'Outside Supported Region' : `India Processing AOI: ${estimation?.area_sqkm ? `${estimation.area_sqkm} km²` : '1.64 km²'}`}</span>
        </div>
      </div>

      {/* 4. REAL-TIME AOI METRICS & SIZE CARD */}
      <div className="p-4 bg-slate-900/95 border-t border-slate-800 space-y-3.5">
        {estimation ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 bg-slate-950/80 rounded-xl border border-slate-800">
            <div>
              <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">AOI Dimensions &amp; Area</div>
              <div className="text-sm font-bold text-sky-400">
                {estimation.dimensions_km.width_km} × {estimation.dimensions_km.height_km} km
              </div>
              <div className="text-[11px] font-medium text-slate-300">
                Area: <span className="text-sky-300 font-semibold">{estimation.area_sqkm} km²</span>
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Sentinel-2 Native Input</div>
              <div className="text-sm font-bold text-slate-100">
                {estimation.native_10m.width} × {estimation.native_10m.height} px
              </div>
              <div className="text-[10px] text-slate-400">
                10m B02, B03, B04, B08 BOA
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">PixelSight SR Output</div>
              <div className="text-sm font-bold text-emerald-400">
                {estimation.super_resolution_2_5m.width} × {estimation.super_resolution_2_5m.height} px
              </div>
              <div className="text-[10px] text-emerald-500/90 font-medium">
                4× super-resolution (~2.5m equivalent)
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Workload &amp; Tiling</div>
              <div className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                <span>{estimation.tiles.total_tiles} tile{estimation.tiles.total_tiles > 1 ? 's' : ''} (128²)</span>
                <span className={`px-1.5 py-0.2 rounded text-[10px] uppercase font-bold ${
                  estimation.category === 'interactive'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                }`}>
                  {estimation.category === 'interactive' ? 'Interactive' : 'Background'}
                </span>
              </div>
              <div className="text-[10px] text-slate-400">
                100 LDSR diffusion steps
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 text-xs text-slate-400 flex items-center gap-2">
            <Info className="w-4 h-4 text-sky-400 flex-shrink-0" />
            <span>Select or move the AOI box inside India to view real-time dimensions and tile estimates.</span>
          </div>
        )}

        {/* Pre-Check Verification Checklist (Rule 18) */}
        <div className="flex flex-wrap items-center gap-3 px-3 py-2 bg-slate-950/60 rounded-xl border border-slate-800 text-[11px]">
          <span className="font-semibold text-slate-400 mr-1 flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-sky-400" /> Pre-Flight Check:
          </span>

          <span className={`flex items-center gap-1 font-medium ${isInsideIndia ? 'text-emerald-400' : 'text-rose-400'}`}>
            {isInsideIndia ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            Inside India
          </span>

          <span className={`flex items-center gap-1 font-medium ${isValidAoi ? 'text-emerald-400' : 'text-rose-400'}`}>
            {isValidAoi ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            Valid AOI
          </span>

          <span className={`flex items-center gap-1 font-medium ${isAreaWithinLimit ? 'text-emerald-400' : 'text-amber-400'}`}>
            {isAreaWithinLimit ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            Area within Limit (≤ 25 km²)
          </span>

          <span className="flex items-center gap-1 font-medium text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Sentinel-2 Compatible (B02/03/04/08)
          </span>

          <span className="flex items-center gap-1 font-medium text-sky-400">
            <CheckCircle2 className="w-3.5 h-3.5" />
            100 LDSR Steps
          </span>
        </div>

        {/* Error Alert if outside India or too large */}
        {estError && (
          <div className="flex items-center gap-2 p-3 bg-rose-500/15 border border-rose-500/40 rounded-xl text-rose-300 text-xs shadow-md">
            <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400" />
            <span className="font-medium">{estError}</span>
          </div>
        )}

        {/* 5. COPERNICUS SEARCH PARAMETERS & TRIGGER ACTION */}
        <div className="flex flex-wrap items-end justify-between gap-4 pt-1">
          <div className="flex flex-wrap items-center gap-3">
            {/* Start Date */}
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1 flex items-center gap-1">
                <Calendar className="w-3 h-3 text-sky-400" /> Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => onStartDateChange(e.target.value)}
                className="px-2.5 py-1.5 bg-slate-950 border border-slate-700/80 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1 flex items-center gap-1">
                <Calendar className="w-3 h-3 text-sky-400" /> End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => onEndDateChange(e.target.value)}
                className="px-2.5 py-1.5 bg-slate-950 border border-slate-700/80 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* Cloud Cover Slider */}
            <div>
              <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 mb-1">
                <span className="flex items-center gap-1">
                  <Cloud className="w-3 h-3 text-sky-400" /> Max Cloud Cover
                </span>
                <span className="text-sky-400">{maxCloudCover}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={maxCloudCover}
                onChange={(e) => onMaxCloudCoverChange(parseFloat(e.target.value))}
                className="w-32 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
              />
            </div>
          </div>

          {/* Search Sentinel-2 Scenes Button */}
          <button
            onClick={onSearchScenes}
            disabled={isSearching || !selectedAoi || !!estError || !isInsideIndia}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-bold rounded-xl shadow-lg shadow-sky-500/20 border border-sky-400/30 transition-all cursor-pointer whitespace-nowrap"
          >
            {isSearching ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Searching Copernicus...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                <span>Search Sentinel-2 Scenes</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
