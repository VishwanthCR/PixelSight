import React, { useState, useEffect } from 'react';
import LandingPage from './pages/LandingPage.jsx';
import ApplicationInputDashboard from './pages/ApplicationInputDashboard.jsx';
import ProcessingPage from './pages/ProcessingPage.jsx';
import ResultsDashboard from './pages/ResultsDashboard.jsx';
import CropMonitoringPage from './pages/CropMonitoringPage.jsx';
import UrbanAnalysisPage from './pages/UrbanAnalysisPage.jsx';
import DisasterManagementPage from './pages/DisasterManagementPage.jsx';
import {
  fetchHealth,
  getJob,
  getReport,
  getResults,
  inspectImage,
  startProcessing,
  startCropProcessing,
  startUrbanProcessing,
  startDisasterProcessing,
} from './api/srmApi.js';

export default function App() {
  const [view, setView] = useState('landing');
  const [application, setApplication] = useState('research');
  const [pipelineStep, setPipelineStep] = useState(null);

  // Single-image states (research, crop, urban)
  const [selectedFile, setSelectedFile] = useState(null);
  const [inspection, setInspection] = useState(null);

  // Dual-image states (disaster)
  const [disasterPreFile, setDisasterPreFile] = useState(null);
  const [disasterPostFile, setDisasterPostFile] = useState(null);
  const [disasterPreInspection, setDisasterPreInspection] = useState(null);
  const [disasterPostInspection, setDisasterPostInspection] = useState(null);

  const [job, setJob] = useState(null);
  const [results, setResults] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

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

  async function handleInspectDisasterPre(file) {
    setError(null);
    setDisasterPreFile(file);
    try {
      setDisasterPreInspection(await inspectImage(file));
    } catch (err) {
      setDisasterPreInspection(null);
      setError(`Pre-event file error: ${err.message}`);
    }
  }

  async function handleInspectDisasterPost(file) {
    setError(null);
    setDisasterPostFile(file);
    try {
      setDisasterPostInspection(await inspectImage(file));
    } catch (err) {
      setDisasterPostInspection(null);
      setError(`Post-event file error: ${err.message}`);
    }
  }

  async function handleProcess() {
    setError(null);
    setView('processing');
    setPipelineStep('queued');

    try {
      let createdJob;
      if (application === 'crop') {
        if (!selectedFile || !inspection?.compatible) return;
        createdJob = await startCropProcessing(selectedFile);
      } else if (application === 'urban') {
        if (!selectedFile || !inspection?.compatible) return;
        createdJob = await startUrbanProcessing(selectedFile);
      } else if (application === 'disaster') {
        if (!disasterPreFile || !disasterPostFile) return;
        createdJob = await startDisasterProcessing(disasterPreFile, disasterPostFile);
      } else {
        if (!selectedFile || !inspection?.compatible) return;
        createdJob = await startProcessing(selectedFile);
      }
      setJob(createdJob);
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
    setDisasterPreFile(null);
    setDisasterPostFile(null);
    setDisasterPreInspection(null);
    setDisasterPostInspection(null);
    setJob(null);
    setResults(null);
    setReport(null);
  }

  const activeApp = job?.application || application;

  return (
    <div style={{ minHeight: '100vh', background: '#050a14' }}>
      {error && (
        <div
          className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-5 py-3 rounded-xl text-sm font-semibold"
          style={{
            background: 'rgba(239,68,68,0.12)',
            border: '1px solid rgba(239,68,68,0.3)',
            color: '#fca5a5',
            backdropFilter: 'blur(16px)',
            whiteSpace: 'nowrap',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          {error}
          <button onClick={() => setError(null)} className="ml-3 opacity-60 hover:opacity-100 text-lg leading-none">
            ✕
          </button>
        </div>
      )}

      {view === 'landing' && (
        <div className="anim-fade-in">
          <LandingPage
            onSelectApplication={(appId) => {
              setApplication(appId);
              setView('dashboard');
            }}
            health={health}
          />
        </div>
      )}

      {view === 'dashboard' && (
        <div className="anim-fade-in">
          <ApplicationInputDashboard
            application={application}
            onBack={() => setView('landing')}
            onJobStarted={(createdJob) => {
              setJob(createdJob);
              setView('processing');
            }}
          />
        </div>
      )}

      {view === 'processing' && (
        <div className="anim-fade-in">
          <ProcessingPage currentStep={pipelineStep} job={job} />
        </div>
      )}

      {view === 'results' && results && report && (
        <div className="anim-fade-in">
          {activeApp === 'crop' && (
            <CropMonitoringPage
              job={job}
              results={results}
              report={report}
              onReset={handleReset}
            />
          )}
          {activeApp === 'urban' && (
            <UrbanAnalysisPage
              job={job}
              results={results}
              report={report}
              onReset={handleReset}
            />
          )}
          {activeApp === 'disaster' && (
            <DisasterManagementPage
              job={job}
              results={results}
              report={report}
              onReset={handleReset}
            />
          )}
          {activeApp === 'research' && (
            <ResultsDashboard
              job={job}
              results={results}
              report={report}
              onReset={handleReset}
            />
          )}
        </div>
      )}
    </div>
  );
}
