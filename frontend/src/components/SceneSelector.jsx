import React from 'react';
import { Calendar, Cloud, Sparkles, Check, CheckCircle, ArrowRight, ShieldCheck } from 'lucide-react';

export default function SceneSelector({
  scenes,
  selectedScene,
  onSelectScene,
  onAutoSelectBest,
  isDisaster = false,
  selectedPreScene,
  onSelectPreScene,
  selectedPostScene,
  onSelectPostScene,
}) {
  if (!scenes || scenes.length === 0) {
    return null;
  }

  const getCloudBadgeColor = (cloud) => {
    if (cloud <= 5) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    if (cloud <= 15) return 'bg-sky-500/10 text-sky-400 border-sky-500/30';
    if (cloud <= 30) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  };

  const formatDate = (isoStr) => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString('en-US', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
      }) + ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
            Available Sentinel-2 Scenes ({scenes.length})
          </h3>
          <p className="text-xs text-slate-400">
            {isDisaster
              ? 'Select two distinct temporal scenes: Pre-Event (baseline) and Post-Event (impacted).'
              : 'Choose a cloud-free observation for PixelSight 4× super-resolution.'}
          </p>
        </div>

        {!isDisaster && onAutoSelectBest && (
          <button
            onClick={onAutoSelectBest}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Auto-Choose Best Scene (Lowest Cloud %)
          </button>
        )}
      </div>

      {/* Disaster Mode Scene Selection Bar */}
      {isDisaster && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-3 bg-slate-950/70 rounded-xl border border-slate-800">
          <div className="space-y-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-sky-400">
              Pre-Event Scene:
            </span>
            <div className="text-xs font-mono text-slate-300 truncate">
              {selectedPreScene ? (
                <span className="text-emerald-400 font-semibold flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> {formatDate(selectedPreScene.datetime)} ({selectedPreScene.cloud_cover}% cloud)
                </span>
              ) : (
                <span className="text-slate-500 italic">No pre-event scene selected</span>
              )}
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-amber-400">
              Post-Event Scene:
            </span>
            <div className="text-xs font-mono text-slate-300 truncate">
              {selectedPostScene ? (
                <span className="text-amber-300 font-semibold flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5" /> {formatDate(selectedPostScene.datetime)} ({selectedPostScene.cloud_cover}% cloud)
                </span>
              ) : (
                <span className="text-slate-500 italic">No post-event scene selected</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Candidate Scenes List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 max-h-72 overflow-y-auto pr-1">
        {scenes.map((scene) => {
          const isSelected = selectedScene?.id === scene.id;
          const isPre = isDisaster && selectedPreScene?.id === scene.id;
          const isPost = isDisaster && selectedPostScene?.id === scene.id;

          return (
            <div
              key={scene.id}
              className={`p-3 rounded-xl border transition-all ${
                isSelected || isPre || isPost
                  ? 'bg-sky-950/30 border-sky-500/70 shadow-md ring-1 ring-sky-500/40'
                  : 'bg-slate-950/50 hover:bg-slate-800/40 border-slate-800'
              }`}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="space-y-0.5">
                  <div className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                    <Calendar className="w-3.5 h-3.5 text-sky-400" />
                    {formatDate(scene.datetime)}
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 truncate max-w-[200px]" title={scene.id}>
                    {scene.id}
                  </div>
                </div>

                <span
                  className={`px-2 py-0.5 rounded-full text-[10px] font-bold border flex items-center gap-1 ${getCloudBadgeColor(
                    scene.cloud_cover
                  )}`}
                >
                  <Cloud className="w-3 h-3" />
                  {scene.cloud_cover}%
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-end gap-2 pt-1 border-t border-slate-800/60">
                {isDisaster ? (
                  <>
                    <button
                      onClick={() => onSelectPreScene(scene)}
                      className={`px-2.5 py-1 text-[11px] font-semibold rounded-md border transition-all ${
                        isPre
                          ? 'bg-sky-500 text-white border-sky-400'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                      }`}
                    >
                      {isPre ? '✓ Pre-Event Selected' : 'Set as Pre-Event'}
                    </button>
                    <button
                      onClick={() => onSelectPostScene(scene)}
                      className={`px-2.5 py-1 text-[11px] font-semibold rounded-md border transition-all ${
                        isPost
                          ? 'bg-amber-500 text-black border-amber-400'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                      }`}
                    >
                      {isPost ? '✓ Post-Event Selected' : 'Set as Post-Event'}
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => onSelectScene(scene)}
                    className={`flex items-center gap-1 px-3 py-1 text-[11px] font-bold rounded-lg transition-all ${
                      isSelected
                        ? 'bg-emerald-500 text-black shadow-sm'
                        : 'bg-sky-600 hover:bg-sky-500 text-white'
                    }`}
                  >
                    {isSelected ? (
                      <>
                        <Check className="w-3 h-3" /> Selected
                      </>
                    ) : (
                      <>Use this scene</>
                    )}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
