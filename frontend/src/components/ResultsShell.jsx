import React from 'react';
import ResultsDashboard from '../pages/ResultsDashboard.jsx';
import CropMonitoringPage from '../pages/CropMonitoringPage.jsx';
import UrbanAnalysisPage from '../pages/UrbanAnalysisPage.jsx';
import DisasterManagementPage from '../pages/DisasterManagementPage.jsx';
import BatchProcessingPage from '../pages/BatchProcessingPage.jsx';
import EvaluationDashboardPage from '../pages/EvaluationDashboardPage.jsx';
import AnalystDrawer from './AnalystDrawer.jsx';

/**
 * ResultsShell & UseCaseRouter
 * ============================
 * Implements Section 31 of PixelSight architecture:
 * The frontend does NOT render a single giant monolithic dashboard with
 * conditional cards. Instead, the authoritative user-selected use case routes
 * directly to its dedicated use-case results page.
 *
 * Each use-case page owns its own visualizations, metrics, and downloads
 * without cross-contamination.
 *
 * PixelSight Analyst drawer is integrated cleanly over the results shell
 * without altering scientific outputs or layout.
 */
export default function ResultsShell({
  job,
  results,
  report,
  onReset,
  onSelectApplication,
  health,
}) {
  const useCase = (
    job?.application ||
    results?.use_case ||
    results?.application ||
    report?.application ||
    'research'
  ).toLowerCase();

  const renderContent = () => {
    switch (useCase) {
      case 'disaster':
        return (
          <DisasterManagementPage
            job={job}
            results={results}
            report={report}
            onReset={onReset}
            onSelectApplication={onSelectApplication}
            health={health}
          />
        );

      case 'crop':
        return (
          <CropMonitoringPage
            job={job}
            results={results}
            report={report}
            onReset={onReset}
            onSelectApplication={onSelectApplication}
            health={health}
          />
        );

      case 'urban':
        return (
          <UrbanAnalysisPage
            job={job}
            results={results}
            report={report}
            onReset={onReset}
            onSelectApplication={onSelectApplication}
            health={health}
          />
        );

      case 'batch':
        return (
          <BatchProcessingPage
            onBack={onReset}
          />
        );

      case 'evaluation':
        return (
          <EvaluationDashboardPage
            onBack={onReset}
          />
        );

      case 'research':
      case 'core':
      default:
        return (
          <ResultsDashboard
            job={job}
            results={results}
            report={report}
            onReset={onReset}
            onSelectApplication={onSelectApplication}
            health={health}
            activeApp="research"
          />
        );
    }
  };

  return (
    <>
      {renderContent()}
      <AnalystDrawer
        job={job}
        results={results}
        report={report}
        application={useCase}
      />
    </>
  );
}

