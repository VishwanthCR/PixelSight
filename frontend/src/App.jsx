import React, { useState, useEffect } from 'react';
import LandingPage from './pages/LandingPage.jsx';
import ProcessingPage from './pages/ProcessingPage.jsx';
import ResultsDashboard from './pages/ResultsDashboard.jsx';
import { fetchHealth, getJob, getReport, getResults, inspectImage, startProcessing } from './api/srmApi.js';

export default function App() {
  const [view,          setView]          = useState('landing');
  const [pipelineStep, setPipelineStep] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [job, setJob] = useState(null);
  const [results, setResults] = useState(null);
  const [report, setReport] = useState(null);
  const [error,         setError]         = useState(null);
  const [health,        setHealth]        = useState(null);

  useEffect(() => {
    let cancelled = false;

    const pollHealth = async () => {
      try {
        const next = await fetchHealth();
        if (!cancelled) setHealth(next);
      } catch {
        if (!cancelled) setHealth({ status: 'offline' });
      }
    };

    pollHealth();
    const timer = window.setInterval(pollHealth, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!job?.job_id || job.status === 'completed' || job.status === 'failed') return undefined;
    const timer = window.setInterval(async () => {
      try {
        const next = await getJob(job.job_id);
        setJob(next);
        setPipelineStep(next.stage);
        if (next.status === 'completed') {
          const [nextResults, nextReport] = await Promise.all([
            getResults(next.job_id),
            getReport(next.job_id),
          ]);
          setResults(nextResults);
          setReport(nextReport);
          setView('results');
        } else if (next.status === 'failed') {
          setError(next.error || 'Processing failed.');
          setView('landing');
        }
      } catch (err) {
        setError(err.message);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function handleInspect(file) {
    setError(null);
    setSelectedFile(file);
    try {
      setInspection(await inspectImage(file));
    } catch (err) {
      setInspection(null);
      setError(err.message);
    }
  }

  async function handleProcess() {
    if (!selectedFile || !inspection?.compatible) return;
    setError(null);
    setView('processing');
    setPipelineStep('queued');
    try {
      setJob(await startProcessing(selectedFile));
    } catch (err) {
      setError(err.message);
      setView('landing');
    }
  }

  function handleReset() {
    setView('landing');
    setPipelineStep(null);
    setSelectedFile(null);
    setInspection(null);
    setJob(null);
    setResults(null);
    setReport(null);
  }

  return (
    <div style={{ minHeight: '100vh', background: '#050a14' }}>
      {error && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-5 py-3 rounded-xl text-sm font-semibold"
          style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5', backdropFilter: 'blur(16px)', whiteSpace: 'nowrap' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          {error}
          <button onClick={() => setError(null)} className="ml-3 opacity-60 hover:opacity-100 text-lg leading-none">✕</button>
        </div>
      )}

      {view === 'landing' && (
        <div className="anim-fade-in">
          <LandingPage file={selectedFile} inspection={inspection} onSelectFile={handleInspect} onProcess={handleProcess} health={health} />
        </div>
      )}

      {view === 'processing' && (
        <div className="anim-fade-in">
          <ProcessingPage currentStep={pipelineStep} job={job} />
        </div>
      )}

      {view === 'results' && results && report && (
        <div className="anim-fade-in">
          <ResultsDashboard
            job={job}
            results={results}
            report={report}
            onReset={handleReset}
          />
        </div>
      )}
    </div>
  );
}
