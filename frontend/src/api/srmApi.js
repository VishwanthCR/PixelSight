const API_BASE_URL = '/api/v1';

export async function fetchHealth() {
  return request('/health');
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json();
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

export function getJob(jobId) {
  return request(`/jobs/${jobId}`);
}

export function getResults(jobId) {
  return request(`/results/${jobId}`);
}

export function getReport(jobId) {
  return request(`/reports/${jobId}`);
}

export function resultFileUrl(jobId, relativePath) {
  return `${API_BASE_URL}/results/${jobId}/files/${relativePath}`;
}
