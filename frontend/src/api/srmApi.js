const API_BASE_URL = '/api/v1';
async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json();
}
export async function fetchHealth() {
  return request('/health');
}
export async function fetchApplications() {
  return request('/applications');
}
export async function fetchCapabilities() {
  return request('/capabilities');
}
export async function inspectImage(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/inspect', { method: 'POST', body: formData });
}
export async function preprocessImage(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/preprocess', { method: 'POST', body: formData });
}
export async function startProcessing(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/process', { method: 'POST', body: formData });
}
export async function startCropProcessing(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/applications/crop', { method: 'POST', body: formData });
}
export async function startUrbanProcessing(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/applications/urban', { method: 'POST', body: formData });
}
export async function startDisasterProcessing(preFile, postFile) {
  const formData = new FormData();
  formData.append('pre_event', preFile);
  formData.append('post_event', postFile);
  return request('/applications/disaster', { method: 'POST', body: formData });
}
export function getJob(jobId) {
  return request(`/jobs/${jobId}`);
}
export function getResults(jobId) {
  return request(`/results/${jobId}`);
}
export function getReport(jobId) {
  return request(`/reports/${jobId}`);
}
export function getManifest(jobId) {
  return request(`/manifest/${jobId}`);
}
export function getApplicationJob(jobId) {
  return request(`/applications/${jobId}`);
}
export function getApplicationResults(jobId) {
  return request(`/applications/${jobId}/results`);
}
export function getApplicationReport(jobId) {
  return request(`/applications/${jobId}/report`);
}
export function resultFileUrl(jobId, relativePath) {
  return `${API_BASE_URL}/results/${jobId}/files/${relativePath}`;
}
export async function fetchCopernicusCapabilities() {
  return request('/copernicus/capabilities');
}
export async function estimateCopernicusAoi(aoi) {
  return request('/copernicus/estimate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ aoi }),
  });
}
export async function searchCopernicusScenes(aoi, startDate, endDate, maxCloudCover = 20.0, limit = 20) {
  return request('/copernicus/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      aoi,
      start_date: startDate,
      end_date: endDate,
      max_cloud_cover: maxCloudCover,
      limit,
    }),
  });
}
export async function acquireCopernicusData(aoi, sceneId, date) {
  return request('/copernicus/acquire', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ aoi, scene_id: sceneId, date }),
  });
}
export async function startCopernicusJob(application, payload) {
  return request(`/applications/${application}/from-copernicus`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
export async function geocodePlace(query) {
  return request(`/copernicus/geocode?q=${encodeURIComponent(query)}`);
}
export async function fetchIndiaBoundary() {
  return request('/copernicus/india-boundary');
}
// ── Compute / Hardware Profile & Execution Plan ──────────────────────────
export async function fetchComputeProfile() {
  return request('/compute/profile');
}
export async function fetchComputePlan(width = 128, height = 128, batchSize = 1, nImages = 1) {
  return request(`/compute/plan?width=${width}&height=${height}&batch_size=${batchSize}&n_images=${nImages}`);
}
// ── Batch Processing & Job Lifecycle ──────────────────────────────────────
export async function startBatchProcessing(files, application = 'research') {
  const formData = new FormData();
  for (const file of files) {
    formData.append('uploads', file);
  }
  formData.append('application', application);
  return request('/batch/process', { method: 'POST', body: formData });
}
export async function listBatchJobs(batchId = '') {
  const qs = batchId ? `?batch_id=${encodeURIComponent(batchId)}` : '';
  return request(`/batch/jobs${qs}`);
}
export async function cancelJob(jobId) {
  return request(`/jobs/${jobId}/cancel`, { method: 'POST' });
}
export async function retryJob(jobId) {
  return request(`/jobs/${jobId}/retry`, { method: 'POST' });
}
// ── Classification & Segmentation ─────────────────────────────────────────
export async function fetchSegmentationStatus() {
  return request('/segmentation/status');
}
export async function fetchClassificationClasses() {
  return request('/classification/classes');
}
export async function fetchJobClassification(jobId) {
  return request(`/results/${jobId}/classification`);
}
export async function startSegmentation(file) {
  const formData = new FormData();
  formData.append('upload', file);
  return request('/segmentation', { method: 'POST', body: formData });
}
// ── Master Evaluation Report ──────────────────────────────────────────────
export async function fetchEvaluationReport() {
  return request('/evaluation/report');
}
// ── Automatic HR Reference Discovery ──────────────────────────────────────
export async function discoverReference(aoi, date = null, metadata = {}) {
  return request('/evaluation/reference/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ aoi, date, metadata }),
  });
}
// ── Ground Truth & Annotation Subsystem ──────────────────────────────────
export async function createGroundTruthWorkspace(jobId, aoi = null, annotator = 'Expert Annotator', notes = '') {
  return request('/ground-truth/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, aoi, annotator, notes }),
  });
}
export async function fetchGroundTruth(jobId) {
  return request(`/ground-truth/${jobId}`);
}
export async function saveGroundTruthAnnotations(jobId, geojson, annotator = 'Expert Annotator', notes = '') {
  return request(`/ground-truth/${jobId}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geojson, annotator, notes }),
  });
}
export async function validateGroundTruth(jobId) {
  return request(`/ground-truth/${jobId}/validate`, { method: 'POST' });
}
export async function rasterizeGroundTruth(jobId, evaluationGrid = '2.5m', autoValidate = false) {
  return request(`/ground-truth/${jobId}/rasterize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ evaluation_grid: evaluationGrid, auto_validate: autoValidate }),
  });
}
export async function reviewGroundTruth(jobId, status, reviewer = null) {
  return request(`/ground-truth/${jobId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, reviewer }),
  });
}
export function groundTruthExportUrl(jobId, format) {
  return `${API_BASE_URL}/ground-truth/${jobId}/export/${format}`;
}

// ── PixelSight Analyst (Local LLM) ────────────────────────────────────────
export async function fetchAnalystStatus() {
  return request('/analyst/status');
}

export async function fetchAnalystModels() {
  return request('/analyst/models');
}

export async function selectAnalystModel(model) {
  return request('/analyst/models/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
}

export async function explainJob(jobId, question = null) {
  return request(`/analyst/jobs/${jobId}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, stream: false }),
  });
}

export async function chatJob(jobId, message) {
  return request(`/analyst/jobs/${jobId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, stream: false }),
  });
}

export async function summarizeBatch(batchId, question = null) {
  return request(`/analyst/batch/${batchId}/summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, stream: false }),
  });
}

export async function clearAnalystSession(sessionId) {
  return request(`/analyst/sessions/${sessionId}`, {
    method: 'DELETE',
  });
}

export async function streamAnalystJob(jobId, message, onChunk, onDone, onError) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyst/jobs/${jobId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
    });
    if (!response.ok) {
      const errPayload = await response.json().catch(() => ({}));
      throw new Error(errPayload.detail || `Stream failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      onChunk(text);
    }
    if (onDone) onDone();
  } catch (err) {
    if (onError) onError(err);
  }
}

