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
