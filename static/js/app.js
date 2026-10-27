const $ = (s) => document.querySelector(s);
const dropzone = $('#dropzone'), input = $('#fileInput'), preview = $('#preview');
const analyzeBtn = $('#analyzeBtn'), errorMsg = $('#errorMsg');
let selectedFiles = [], chart = null, resultData = null;

function setFiles(files) {
  errorMsg.textContent = '';
  const incoming = Array.from(files || []);
  if (!incoming.length || incoming.some(file => !file.type.startsWith('image/'))) return showError('Please choose JPG, PNG or WebP images.');
  if (incoming.length > 50) return showError('A batch can contain at most 50 images.');
  if (incoming.some(file => file.size > 10 * 1024 * 1024)) return showError('Each image must be smaller than 10 MB.');
  selectedFiles = incoming;
  preview.src = URL.createObjectURL(incoming[0]); preview.hidden = false;
  $('#emptyUpload').hidden = true; $('#removeImage').hidden = false; analyzeBtn.disabled = false;
  $('#batchSummary').textContent = incoming.length > 1 ? `${incoming.length} images ready for batch analysis.` : '';
}
function showError(message) { errorMsg.textContent = message; }
dropzone.addEventListener('click', (e) => { if (!e.target.closest('#removeImage')) input.click(); });
dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
input.addEventListener('change', () => setFiles(input.files));
['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, e => {e.preventDefault();dropzone.classList.add('dragover')}));
['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {e.preventDefault();dropzone.classList.remove('dragover')}));
dropzone.addEventListener('drop', e => setFiles(e.dataTransfer.files));
$('#removeImage').addEventListener('click', e => {e.stopPropagation(); selectedFiles=[]; input.value=''; preview.hidden=true; $('#emptyUpload').hidden=false; e.currentTarget.hidden=true; analyzeBtn.disabled=true; $('#batchSummary').textContent='';});
$('#cameraBtn').addEventListener('click', () => input.click());

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  $('#idlePanel').hidden = true; $('#resultsPanel').hidden = true; $('#loadingPanel').hidden = false;
  analyzeBtn.disabled = true; errorMsg.textContent = '';
  const form = new FormData();
  form.append('fruit_type', $('#fruitType').value); form.append('temperature_c', $('#temperature').value); form.append('humidity_pct', $('#humidity').value);
  const isBatch = selectedFiles.length > 1;
  selectedFiles.forEach(file => form.append(isBatch ? 'images' : 'image', file));
  try {
    const response = await fetch(isBatch ? '/api/analyze-batch' : '/api/analyze', {method:'POST', body:form});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    resultData = isBatch ? data.results[0] : data;
    if (!resultData) throw new Error('No images in the batch could be analyzed.');
    if (isBatch) $('#batchSummary').textContent = `Batch complete: ${data.count} analyzed, ${data.failed} failed. Showing the first result.`;
    renderResult(resultData);
  } catch (err) { showError(err.message); $('#idlePanel').hidden=false; }
  finally { $('#loadingPanel').hidden=true; analyzeBtn.disabled=false; }
});

function renderResult(data) {
  $('#resultFruit').textContent = data.fruit;
  $('#gradeBadge').textContent = `Grade ${data.grade}`;
  $('#statusBadge').textContent = data.status;
  $('#freshnessScore').textContent = Math.round(data.freshness_score);
  $('#scoreRing').style.background = `conic-gradient(var(--lime) ${data.freshness_score*3.6}deg,#23372b 0deg)`;
  $('#shelfDays').textContent = data.shelf_life_days ?? 'Not available';
  $('#confidenceText').textContent = data.model_mode === 'visual_prototype' ? 'Train and install a checkpoint for model confidence' : `${data.confidence}% analysis confidence`;
  $('#recommendation').textContent = data.recommendation;
  $('#disclaimer').textContent = data.disclaimer;
  const gradcam = $('#gradcamLink'); gradcam.hidden = !data.gradcam_url; if(data.gradcam_url) gradcam.href = data.gradcam_url;
  $('#resultsPanel').hidden = false; showMetrics();
}
function baseOptions(max=100) { return {responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:{color:'#82958a',font:{size:10}}},y:{min:0,max,grid:{color:'#ffffff0c'},ticks:{color:'#82958a',font:{size:9}}}}}; }
function showMetrics() {
  const entries = Object.entries(resultData.metrics);
  if(chart) chart.destroy();
  chart = new Chart($('#qualityChart'), {type:'bar',data:{labels:entries.map(x=>x[0].replace('Estimated ','')),datasets:[{data:entries.map(x=>x[1]),backgroundColor:entries.map((_,i)=>i===3?'#ff9d5c':'#b9f227'),borderRadius:5,barThickness:22}]},options:baseOptions()});
}
function showForecast() {
  if(chart) chart.destroy();
  if(!resultData.forecast.length) return;
  chart = new Chart($('#qualityChart'), {type:'line',data:{labels:resultData.forecast.map(x=>`Day ${x.day}`),datasets:[{data:resultData.forecast.map(x=>x.freshness),borderColor:'#b9f227',backgroundColor:'#b9f22718',fill:true,tension:.38,pointBackgroundColor:'#b9f227',pointRadius:3}]},options:baseOptions()});
}
document.querySelectorAll('.chart-tabs button').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.chart-tabs button').forEach(b=>b.classList.remove('active')); btn.classList.add('active');
  btn.dataset.chart === 'metrics' ? showMetrics() : showForecast();
}));
