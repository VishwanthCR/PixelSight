import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles, Bot, User, Send, RefreshCw, X, ChevronRight,
  AlertTriangle, ShieldCheck, HelpCircle, Terminal, CheckCircle2,
  Trash2, MessageSquare, ExternalLink
} from 'lucide-react';
import {
  fetchAnalystStatus,
  fetchAnalystModels,
  selectAnalystModel,
  explainJob,
  chatJob,
  streamAnalystJob,
  summarizeBatch,
  clearAnalystSession,
} from '../api/srmApi.js';

// Simple lightweight markdown parser for scientific responses
function FormattedMessage({ content }) {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeBuffer = [];

  lines.forEach((line, index) => {
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`code-${index}`} className="my-2 p-2.5 rounded bg-black/60 border border-white/10 text-xs font-mono text-cyan-300 overflow-x-auto">
            <code>{codeBuffer.join('\n')}</code>
          </pre>
        );
        codeBuffer = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      return;
    }

    // Headers
    if (line.startsWith('### ')) {
      elements.push(<h4 key={index} className="text-sm font-semibold text-cyan-300 mt-3 mb-1">{line.slice(4)}</h4>);
      return;
    }
    if (line.startsWith('## ')) {
      elements.push(<h3 key={index} className="text-sm font-bold text-slate-100 mt-3 mb-1.5">{line.slice(3)}</h3>);
      return;
    }

    // Bullet points
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      const text = line.trim().slice(2);
      elements.push(
        <div key={index} className="flex items-start gap-2 my-1 text-xs text-slate-300 leading-relaxed">
          <span className="text-cyan-400 mt-1 text-[8px]">●</span>
          <span>{renderInline(text)}</span>
        </div>
      );
      return;
    }

    // Numbered lists
    const numMatch = line.trim().match(/^(\d+)\.\s+(.*)$/);
    if (numMatch) {
      elements.push(
        <div key={index} className="flex items-start gap-2 my-1 text-xs text-slate-300 leading-relaxed">
          <span className="font-mono text-cyan-400 text-xs">{numMatch[1]}.</span>
          <span>{renderInline(numMatch[2])}</span>
        </div>
      );
      return;
    }

    // Empty lines
    if (!line.trim()) {
      elements.push(<div key={index} className="h-1.5" />);
      return;
    }

    // Standard paragraph
    elements.push(
      <p key={index} className="text-xs text-slate-300 leading-relaxed my-1">
        {renderInline(line)}
      </p>
    );
  });

  return <div className="space-y-0.5">{elements}</div>;
}

// Inline formatting (bold, code, quotes)
function renderInline(text) {
  const parts = [];
  let remaining = text;
  let key = 0;

  // Pattern for **bold** and `code`
  const regex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let match;
  let lastIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(<span key={key++}>{text.substring(lastIdx, match.index)}</span>);
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={key++} className="font-semibold text-slate-100">{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code key={key++} className="px-1 py-0.5 rounded bg-black/40 border border-white/10 font-mono text-[11px] text-cyan-200">
          {token.slice(1, -1)}
        </code>
      );
    }
    lastIdx = regex.lastIndex;
  }
  if (lastIdx < text.length) {
    parts.push(<span key={key++}>{text.substring(lastIdx)}</span>);
  }
  return parts.length ? parts : text;
}

export default function AnalystDrawer({
  job,
  report,
  results,
  batchId = null,
  application = 'research',
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const activeJobId = job?.job_id;
  const activeApp = (job?.application || application || 'research').toLowerCase();

  // Scroll messages to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Load Ollama status on mount
  const checkStatus = async () => {
    try {
      const s = await fetchAnalystStatus();
      setStatus(s);
      if (s.ollama_available) {
        const m = await fetchAnalystModels();
        setModels(m.models || []);
      }
    } catch {
      setStatus({
        ollama_available: false,
        model_available: false,
        status: 'ollama_offline',
        error: 'PixelSight Analyst is unavailable because the local LLM service is not running.',
        instructions: 'Start Ollama by running `ollama serve` or launching the Ollama desktop app.',
      });
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  // When job changes, clear local session or notify
  useEffect(() => {
    setMessages([]);
    setError(null);
  }, [activeJobId, batchId]);

  // Model change handler
  const handleModelChange = async (newModel) => {
    try {
      await selectAnalystModel(newModel);
      await checkStatus();
    } catch (err) {
      setError(err.message);
    }
  };

  // Clear chat
  const handleClear = async () => {
    const sessId = activeJobId || batchId;
    if (sessId) {
      try {
        await clearAnalystSession(sessId);
      } catch {
        // silent
      }
    }
    setMessages([]);
    setError(null);
  };

  // Send message
  const handleSend = async (textToSend) => {
    const text = (textToSend || input).trim();
    if (!text || loading) return;

    setInput('');
    setError(null);

    // Append user message
    const userMsg = { role: 'user', content: text, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      if (batchId) {
        // Batch summary / chat
        const res = await summarizeBatch(batchId, text);
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: res.message, timestamp: res.timestamp },
        ]);
      } else if (activeJobId) {
        // Try streaming first, fallback to regular chat
        let streamedText = '';
        let assistantAdded = false;

        await streamAnalystJob(
          activeJobId,
          text,
          (chunk) => {
            streamedText += chunk;
            if (!assistantAdded) {
              assistantAdded = true;
              setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: streamedText, timestamp: new Date().toISOString() },
              ]);
            } else {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  ...next[next.length - 1],
                  content: streamedText,
                };
                return next;
              });
            }
          },
          () => {
            setLoading(false);
          },
          async () => {
            // Streaming fallback to non-streaming POST
            const fallback = await chatJob(activeJobId, text);
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: fallback.message, timestamp: fallback.timestamp },
            ]);
            setLoading(false);
          }
        );
      } else {
        throw new Error('No active job or batch selected to analyze.');
      }
    } catch (err) {
      setError(err.message || 'Error communicating with PixelSight Analyst.');
      setLoading(false);
    }
  };

  // Quick Action triggers
  const handleExplain = async () => {
    if (!activeJobId && !batchId) return;
    setLoading(true);
    setError(null);
    try {
      if (batchId) {
        const res = await summarizeBatch(batchId);
        setMessages((prev) => [
          ...prev,
          { role: 'user', content: 'Explain Batch Results', timestamp: new Date().toISOString() },
          { role: 'assistant', content: res.message, timestamp: res.timestamp },
        ]);
      } else {
        const res = await explainJob(activeJobId);
        setMessages((prev) => [
          ...prev,
          { role: 'user', content: 'Explain Job Results', timestamp: new Date().toISOString() },
          { role: 'assistant', content: res.message, timestamp: res.timestamp },
        ]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Determine available quick actions based on real outputs
  const getQuickActions = () => {
    if (batchId) {
      return [
        { label: 'Summarize Batch', prompt: 'Summarize the overall throughput, success/failure rate, and performance of this batch run.' },
        { label: 'Explain Failed Jobs', prompt: 'Identify and explain any failed jobs or processing errors in this batch.' },
        { label: 'Compare Job Results', prompt: 'Compare reconstruction quality and uncertainty across the processed tiles.' },
      ];
    }

    const evalAvailable = report?.evaluation?.status === 'reference_available';
    const uncAvailable = !!(report?.uncertainty || results?.outputs?.uncertainty);

    if (activeApp === 'crop') {
      const actions = [
        { label: 'Explain NDVI Results', prompt: 'Explain the NDVI distribution and canopy preservation between native Sentinel-2 and the super-resolved representation.' },
        { label: 'Explain Spectral Preservation', prompt: 'How well did the LDSR-S2 super-resolution preserve spectral fidelity across VNIR bands?' },
      ];
      if (uncAvailable) actions.push({ label: 'Explain Uncertainty', prompt: 'What does the stochastic diffusion uncertainty map indicate for this agricultural AOI?' });
      actions.push({ label: 'Explain Limitations', prompt: 'What are the scientific limitations of this crop analysis?' });
      return actions;
    }

    if (activeApp === 'urban') {
      const actions = [
        { label: 'Explain Segmentation', prompt: 'Explain the urban land-cover segmentation results and class distribution.' },
        { label: 'Explain IoU/F1', prompt: 'Explain the segmentation accuracy metrics (mIoU and Dice/F1) and how they relate to the super-resolved output.' },
      ];
      if (uncAvailable) actions.push({ label: 'Explain Uncertainty', prompt: 'How does diffusion uncertainty correlate with complex urban structures and boundaries?' });
      actions.push({ label: 'Explain Class Behavior', prompt: 'Which urban classes exhibited the highest and lowest agreement?' });
      return actions;
    }

    if (activeApp === 'disaster') {
      const actions = [
        { label: 'Explain Detected Changes', prompt: 'Explain the observed surface differences between pre-event and post-event imagery.' },
        { label: 'Explain Change Statistics', prompt: 'Break down the change statistics and affected area estimations.' },
      ];
      if (uncAvailable) actions.push({ label: 'Explain Uncertainty', prompt: 'What does the uncertainty analysis show regarding the change detection confidence?' });
      actions.push({ label: 'Explain Limitations', prompt: 'What are the scientific limitations in distinguishing spectral changes from validated physical disaster damage?' });
      return actions;
    }

    // Default / Core / Research
    const actions = [
      { label: 'Explain Reconstruction Quality', prompt: 'Explain the LDSR-S2 4x super-resolution reconstruction quality for this scene.' },
    ];
    if (evalAvailable) {
      actions.push({ label: 'Explain Evaluation', prompt: 'Explain the PSNR, SSIM, and SAM metrics computed against the external HR reference.' });
    } else {
      actions.push({ label: 'Why Are HR Metrics Withheld?', prompt: 'Why are PSNR, SSIM, and SAM withheld, and what no-reference diagnostics were provided instead?' });
    }
    if (uncAvailable) {
      actions.push({ label: 'Explain Uncertainty', prompt: 'Explain what the stochastic diffusion variance and high-uncertainty regions signify.' });
    }
    actions.push({ label: 'Explain Limitations', prompt: 'What are the core scientific and methodological limitations of this result?' });
    return actions;
  };

  const isReady = status?.status === 'ready';
  const isOffline = status?.status === 'ollama_offline';
  const isModelMissing = status?.status === 'model_missing';

  return (
    <>
      {/* Floating Launcher Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-3 px-4 py-3 rounded-2xl shadow-2xl transition-all duration-300 hover:scale-105"
          style={{
            background: 'linear-gradient(135deg, rgba(8,145,178,0.25) 0%, rgba(15,23,42,0.9) 100%)',
            border: '1px solid rgba(6,182,212,0.4)',
            backdropFilter: 'blur(16px)',
            boxShadow: '0 10px 30px -5px rgba(6,182,212,0.3)',
          }}
          title="Open PixelSight Analyst"
        >
          <div className="relative">
            <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse" />
            <span
              className={`absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full ring-2 ring-[#050a14] ${
                isReady ? 'bg-emerald-400' : isModelMissing ? 'bg-amber-400' : 'bg-red-400'
              }`}
            />
          </div>
          <div className="text-left">
            <div className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
              PixelSight Analyst
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-800/40">
                LOCAL
              </span>
            </div>
            <div className="text-[10px] text-slate-400">
              {isReady
                ? status.configured_model
                : isModelMissing
                ? 'Model Missing'
                : 'Ollama Offline'}
            </div>
          </div>
        </button>
      )}

      {/* Slide-out Analyst Drawer */}
      {isOpen && (
        <div
          className="fixed top-0 right-0 bottom-0 z-50 flex flex-col shadow-2xl transition-transform duration-300"
          style={{
            width: '460px',
            maxWidth: '100vw',
            background: 'rgba(7, 12, 22, 0.95)',
            borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(24px)',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-slate-950/50">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <Bot className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-slate-100">PixelSight Analyst</h3>
                  <span
                    className={`inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full ${
                      isReady
                        ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                        : isModelMissing
                        ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                        : 'bg-red-500/15 text-red-300 border border-red-500/30'
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        isReady ? 'bg-emerald-400' : isModelMissing ? 'bg-amber-400' : 'bg-red-400'
                      }`}
                    />
                    {isReady ? 'Ready' : isModelMissing ? 'Model Missing' : 'Offline'}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400">
                  Local Instruct LLM · Zero Cloud Dependency
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={handleClear}
                title="Clear chat history"
                className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Model Selector Bar */}
          <div className="flex items-center justify-between px-5 py-2.5 bg-black/30 border-b border-white/[0.05] text-[11px]">
            <span className="text-slate-400 flex items-center gap-1">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              Model:
            </span>
            {models.length > 1 ? (
              <select
                value={status?.configured_model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="bg-slate-900 border border-white/10 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-cyan-500"
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <span className="font-mono text-cyan-300 font-semibold">
                {status?.configured_model || 'qwen2.5:7b'}
              </span>
            )}
          </div>


          {/* Offline / Error Banner */}
          {!isReady && (
            <div className="p-4 mx-4 mt-4 rounded-xl border border-amber-500/25 bg-amber-500/[0.06] text-amber-200/90 text-xs">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="space-y-1.5 flex-1">
                  <div className="font-semibold text-amber-300">
                    {isOffline
                      ? 'Local LLM Service Offline'
                      : isModelMissing
                      ? 'Configured Model Missing'
                      : 'Ollama Warning'}
                  </div>
                  <p className="text-[11px] leading-relaxed text-amber-200/80">
                    {status?.error || 'Could not connect to Ollama runtime.'}
                  </p>
                  {status?.instructions && (
                    <div className="p-2 rounded bg-black/40 border border-amber-500/20 font-mono text-[10px] text-amber-100">
                      {status.instructions}
                    </div>
                  )}
                  <div className="pt-1 flex items-center justify-between">
                    <span className="text-[10px] text-slate-400">
                      PixelSight core processing continues unaffected.
                    </span>
                    <button
                      onClick={checkStatus}
                      className="flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300 font-semibold"
                    >
                      <RefreshCw className="w-3 h-3" /> Retry
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Scientific Disclaimer Card */}
          <div className="px-5 py-2 text-[11px] text-slate-500 flex items-center gap-1.5 border-b border-white/[0.04]">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-500/70" />
            <span>Authoritative measurements computed by PixelSight. Analyst explains data.</span>
          </div>

          {/* Quick Action Chips */}
          <div className="px-5 py-3 border-b border-white/[0.05] bg-slate-950/20">
            <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-2">
              Context Actions ({activeApp.toUpperCase()})
            </div>
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={handleExplain}
                disabled={loading || !isReady}
                className="px-2.5 py-1 rounded-lg text-xs font-medium bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-300 transition-colors disabled:opacity-40"
              >
                ✨ Explain Results
              </button>
              {getQuickActions().map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(action.prompt)}
                  disabled={loading || !isReady}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-slate-300 hover:text-slate-100 transition-colors disabled:opacity-40"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
                <Bot className="w-10 h-10 mb-3 text-cyan-500/40" />
                <h4 className="text-sm font-semibold text-slate-300 mb-1">
                  Ready to explain this {activeApp} result
                </h4>
                <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                  Click <strong className="text-cyan-400">Explain Results</strong> or select any action above to interpret super-resolution, uncertainty, and metrics.
                </p>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'assistant' && (
                    <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400 flex-shrink-0 mt-0.5">
                      <Sparkles className="w-3.5 h-3.5" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed ${
                      msg.role === 'user'
                        ? 'bg-cyan-600 text-white rounded-tr-sm shadow-md'
                        : 'bg-white/[0.04] border border-white/[0.08] text-slate-200 rounded-tl-sm shadow-inner'
                    }`}
                  >
                    {msg.role === 'user' ? (
                      <div className="font-medium whitespace-pre-wrap">{msg.content}</div>
                    ) : (
                      <FormattedMessage content={msg.content} />
                    )}
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-7 h-7 rounded-lg bg-slate-800 border border-white/10 flex items-center justify-center text-slate-300 flex-shrink-0 mt-0.5">
                      <User className="w-3.5 h-3.5" />
                    </div>
                  )}
                </div>
              ))
            )}

            {/* Typing / Loading indicator */}
            {loading && (
              <div className="flex gap-3 justify-start items-center">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400 flex-shrink-0">
                  <Sparkles className="w-3.5 h-3.5 animate-spin" />
                </div>
                <div className="bg-white/[0.04] border border-white/[0.08] rounded-2xl px-4 py-2 text-xs text-cyan-300 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  <span>Synthesizing scientific interpretation...</span>
                </div>
              </div>
            )}

            {error && (
              <div className="p-3 rounded-lg border border-red-500/30 bg-red-500/10 text-red-200 text-xs">
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input Bar */}
          <div className="p-4 border-t border-white/[0.08] bg-slate-950/60">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="flex items-center gap-2"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isReady ? 'Ask about this result...' : 'Ollama not ready...'}
                disabled={loading || !isReady}
                className="flex-1 bg-white/[0.04] border border-white/10 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={loading || !input.trim() || !isReady}
                className="p-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition-all disabled:opacity-30 disabled:hover:bg-cyan-500 shadow-md"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <div className="mt-2 text-[10px] text-center text-slate-500">
              Answers are grounded strictly in PixelSight job metrics · No cloud transmission
            </div>
          </div>
        </div>
      )}
    </>
  );
}
