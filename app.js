import { createIcons, Settings, FileUp, ArrowUp, BookOpen, X } from 'https://cdn.jsdelivr.net/npm/lucide@0.468.0/dist/esm/lucide.js';

const $ = (s) => document.querySelector(s);
const ui = { settings: $('#settingsDialog'), question: $('#questionInput'), messages: $('#messages'), toast: $('#toast'), pdf: $('#pdfInput') };
const state = { index: '', namespace: '', busy: false };

function toast(message) {
  ui.toast.textContent = message; ui.toast.classList.add('show');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => ui.toast.classList.remove('show'), 3200);
}
function setProgress(percent, message) {
  $('#uploadProgress').hidden = false; $('#progressFill').style.width = `${percent}%`; $('#progressText').textContent = message;
}
function escapeHtml(text) { return String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function formatAnswer(text) {
  return escapeHtml(text.trim()).split(/\n{2,}/).map(p => `<p>${p.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</p>`).join('');
}
function renderMessage(role, text, docs = []) {
  const row = document.createElement('div'); row.className = `message ${role}`;
  if (role === 'user') row.innerHTML = `<div class="message-body"><div class="user-bubble">${escapeHtml(text)}</div></div>`;
  else {
    const citations = [...new Set(docs.map(d => d.page).filter(p => p != null))].sort((a,b)=>a-b);
    row.innerHTML = `<div class="message-avatar">P</div><div class="message-body"><div class="answer-content">${formatAnswer(text)}</div>${citations.length ? `<div class="sources"><span class="sources-label">Sources</span>${citations.map(p=>`<span class="source-chip">Page ${Number(p)+1}</span>`).join('')}</div>` : ''}</div>`;
  }
  ui.messages.append(row); $('#chatScroll').scrollTop = $('#chatScroll').scrollHeight; return row;
}
async function requestJson(url, options = {}) {
  let response;
  try { response = await fetch(url, options); }
  catch { throw new Error('Could not reach the Python server. Start it with uvicorn main:app --reload.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status}).`);
  return body;
}
function setActiveIndex(index, namespace) {
  state.index = index; state.namespace = namespace;
  $('#activeDbLabel').textContent = index + (namespace && namespace !== index ? ` · ${namespace}` : '');
  $('#activeLibrary').hidden = false;
}
async function refreshBackendStatus() {
  $('#backendStatus').textContent = 'Checking backend…';
  try {
    const health = await requestJson('/api/health');
    const missing = [];
    if (!health.nvidia_configured) missing.push('NVIDIA_API_KEY');
    if (!health.pinecone_configured) missing.push('PINECONE_API_KEY');
    $('#backendStatus').textContent = missing.length ? `Add ${missing.join(' and ')} to .env, then restart the server.` : 'Python backend is connected. Provider keys are loaded from .env.';
    $('#connectionState').classList.toggle('connected', missing.length === 0);
  } catch (e) {
    $('#backendStatus').textContent = e.message;
    $('#connectionState').classList.remove('connected');
  }
}
async function processPdf() {
  const file = ui.pdf.files[0]; const name = $('#uploadIndexName').value.trim().toLowerCase(); $('#uploadError').textContent = '';
  if (!file) { $('#uploadError').textContent = 'Choose a PDF first.'; return; }
  if (!/^[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?$/.test(name)) { $('#uploadError').textContent = 'Use 1–45 lowercase letters, numbers, and hyphens.'; return; }
  $('#startUpload').disabled = true; setProgress(8, 'Processing PDF…'); $('#uploadProgress').classList.add('processing');
  try {
    const form = new FormData(); form.append('file', file); form.append('index_name', name); form.append('namespace', $('#namespace').value.trim());
    setProgress(24, 'Reading and embedding PDF…');
    const result = await requestJson('/api/upload', { method: 'POST', body: form });
    setProgress(100, `${result.chunks} passages indexed`); setActiveIndex(result.index_name, result.namespace);
    toast('PDF added to vector database.');
    $('#uploadProgress').classList.remove('processing');
    setTimeout(() => { $('#uploadProgress').hidden = true; $('#progressFill').style.width = '0'; $('#startUpload').disabled = false; }, 1200);
  } catch (e) { $('#uploadError').textContent = e.message; $('#startUpload').disabled = false; $('#uploadProgress').hidden = true; $('#uploadProgress').classList.remove('processing'); }
}
async function connectIndex() {
  const name = $('#indexName').value.trim().toLowerCase(); $('#indexError').textContent = '';
  if (!name) { $('#indexError').textContent = 'Enter a database name.'; return; }
  $('#librarySubmit').disabled = true; $('#librarySubmit').textContent = 'Loading…';
  try {
    const result = await requestJson('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ index_name: name, namespace: $('#namespace').value.trim() }) });
    setActiveIndex(result.index_name, result.namespace); toast(`Loaded ${name}.`);
  } catch (e) { $('#indexError').textContent = e.message; }
  finally { $('#librarySubmit').disabled = false; $('#librarySubmit').textContent = 'Load database'; }
}
async function answerQuestion(question) {
  if (!state.index) { toast('Load a vector database first.'); $('#indexName').focus(); return; }
  state.busy = true; $('#sendButton').disabled = true; renderMessage('user', question);
  const waiting = document.createElement('div'); waiting.className = 'message'; waiting.innerHTML = '<div class="message-avatar">P</div><div class="message-body typing">Thinking</div>'; ui.messages.append(waiting);
  try {
    const result = await requestJson('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ index_name: state.index, namespace: state.namespace, question }) });
    waiting.remove(); renderMessage('assistant', result.answer, result.sources || []);
  } catch (e) { waiting.remove(); renderMessage('assistant', e.message); }
  finally { state.busy = false; $('#sendButton').disabled = false; ui.question.focus(); }
}

$('#settingsOpen').addEventListener('click', () => { ui.settings.showModal(); refreshBackendStatus(); });
$('#saveSettings').addEventListener('click', () => {
  if (state.index) setActiveIndex(state.index, $('#namespace').value.trim() || state.index);
  toast('Namespace saved.');
});
$('#choosePdf').addEventListener('click', () => ui.pdf.click());
ui.pdf.addEventListener('change', () => {
  $('#fileName').textContent = ui.pdf.files[0]?.name || 'Choose a PDF';
  if (!$('#uploadIndexName').value && ui.pdf.files[0]) $('#uploadIndexName').value = ui.pdf.files[0].name.replace(/\.pdf$/i, '').toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 45) || 'my-library';
});
$('#startUpload').addEventListener('click', processPdf); $('#librarySubmit').addEventListener('click', connectIndex);
$('#chatForm').addEventListener('submit', e => { e.preventDefault(); const question = ui.question.value.trim(); if (!question || state.busy) return; ui.question.value = ''; ui.question.style.height = 'auto'; answerQuestion(question); });
ui.question.addEventListener('input', () => { ui.question.style.height = 'auto'; ui.question.style.height = `${Math.min(ui.question.scrollHeight, 145)}px`; });
ui.question.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#chatForm').requestSubmit(); } });
$('#clearDb').addEventListener('click', () => { state.index = ''; state.namespace = ''; $('#activeLibrary').hidden = true; });
updateConnectionState();
function updateConnectionState() { refreshBackendStatus(); }
createIcons({ icons: { Settings, FileUp, ArrowUp, BookOpen, X } });
