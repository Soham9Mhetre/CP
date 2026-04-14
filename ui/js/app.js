/* ===================================================================
   app.js — CryptoGuard Blockchain Fraud Prevention Dashboard
   Elliptic Bitcoin Dataset · FraudGAT (Spectral + GAT + LSTM)
   =================================================================== */

const API = '';   // same origin

// ── Utility ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function formatBTC(n) {
  return n >= 0.001 ? n.toFixed(6) + ' BTC' : (n * 1e8).toFixed(0) + ' sat';
}

function timeAgo(ts) {
  const diff = Math.floor(Date.now() / 1000) - ts;
  if (diff < 60)   return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function riskIcon(risk) {
  if (risk === 'HIGH')   return '🚨';
  if (risk === 'MEDIUM') return '⚠️';
  return '✅';
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'success', duration = 3500) {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type==='success'?'✓':type==='error'?'✕':'⚠'}</span> ${msg}`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Animated count-up ─────────────────────────────────────────────────────────
function animateCount(el, target, decimals = 0, prefix = '', suffix = '', duration = 1200) {
  if (!el) return;
  let startTime = null;
  const step = ts => {
    if (!startTime) startTime = ts;
    const prog = Math.min((ts - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - prog, 3);
    el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
    if (prog < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Chart instances ───────────────────────────────────────────────────────────
let timelineChart = null;
let distChart     = null;

// ── API fetch helper ──────────────────────────────────────────────────────────
async function apiFetch(url) {
  const res = await fetch(API + url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Health / system status ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const d = await apiFetch('/api/health');
    const el = $('status-text');
    if (el) el.textContent = d.status === 'healthy' ? 'Prevention Active' : 'System Offline';
    const mm = $('model-mode');
    if (mm) mm.textContent = `🧠 ${d.mode === 'simulation' ? 'Simulation Mode' : 'Model Active'}`;
  } catch { /* silent */ }
}

// ── Load + display Elliptic stats ─────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await apiFetch('/api/stats');
    // KPI cards  (Elliptic numbers)
    animateCount($('kpi-total'),  203769, 0);
    animateCount($('kpi-fraud'),    4545, 0);
    animateCount($('kpi-recall'),     68, 0, '', '%');
    animateCount($('kpi-acc'),        89, 0, '', '%');

    // Analytics metrics (from README)
    const acc = $('stat-accuracy');  if (acc) acc.textContent = '89%';
    const rec = $('stat-recall');    if (rec) rec.textContent = '68%';
    const pre = $('stat-precision'); if (pre) pre.textContent = '33%';
    const f1  = $('stat-f1');        if (f1)  f1.textContent  = '45%';

  } catch (err) {
    console.error('Stats load failed:', err);
    // Fallback statics
    if ($('kpi-total'))  $('kpi-total').textContent  = '203,769';
    if ($('kpi-fraud'))  $('kpi-fraud').textContent  = '4,545';
    if ($('kpi-recall')) $('kpi-recall').textContent = '68%';
    if ($('kpi-acc'))    $('kpi-acc').textContent    = '89%';
  }
}

// ── Transaction Feed ──────────────────────────────────────────────────────────
async function loadFeed() {
  try {
    const txns = await apiFetch('/api/sample-transactions?n=18');
    renderFeed(txns);
  } catch (err) { console.error('Feed load failed:', err); }
}

function renderFeed(txns) {
  const feed = $('txn-feed');
  if (!feed) return;
  feed.innerHTML = '';

  txns.forEach((t, i) => {
    const item = document.createElement('div');
    item.className = 'txn-item';
    item.style.animationDelay = `${i * 0.04}s`;

    const scoreColor =
      t.risk === 'HIGH'   ? 'var(--accent-red)'   :
      t.risk === 'MEDIUM' ? 'var(--accent-amber)'  : 'var(--accent-cyan)';

    item.innerHTML = `
      <div class="txn-risk-badge risk-${t.risk}">${riskIcon(t.risk)}</div>
      <div class="txn-info">
        <div class="txn-id">${t.id}</div>
        <div class="txn-meta">${t.entity} · ${formatBTC(t.btc)} · Step T${t.time_step}</div>
        <div class="txn-meta" style="font-family:monospace;font-size:10px;color:#4b5563">${t.wallet}</div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div class="txn-score" style="color:${scoreColor}">${(t.fraud_prob * 100).toFixed(1)}%</div>
        <div class="txn-decision decision-${t.decision}">${t.decision}</div>
      </div>
    `;
    feed.appendChild(item);
  });
}

let _feedInterval = null;
function startFeedRefresh() {
  clearInterval(_feedInterval);
  _feedInterval = setInterval(loadFeed, 8000);
}

// ── Timeline chart ─────────────────────────────────────────────────────────────
async function loadTimelineChart() {
  try {
    const data = await apiFetch('/api/timeline');
    const ctx  = $('timelineChart');
    if (!ctx) return;
    if (timelineChart) timelineChart.destroy();

    timelineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => `T${d.time_step}`),
        datasets: [{
          label: 'Illicit Rate (%)',
          data: data.map(d => d.fraud_rate),
          borderColor: '#ff3864',
          backgroundColor: 'rgba(255,56,100,0.08)',
          fill: true, tension: 0.4, pointRadius: 0,
          pointHoverRadius: 5, pointHoverBackgroundColor: '#ff3864',
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(12,18,33,0.95)',
            borderColor:     'rgba(255,56,100,0.35)',
            borderWidth: 1,
            titleColor: '#e2e8f0',
            bodyColor:  '#ff3864',
            padding: 10,
            callbacks: {
              label: c => ` Illicit Rate: ${c.parsed.y.toFixed(2)}%`,
              title: ts => `Time Step ${ts[0].label}`,
            }
          }
        },
        scales: {
          x: { grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#64748b',maxTicksLimit:12,font:{size:11}}, border:{color:'transparent'} },
          y: { grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#64748b',font:{size:11},callback:v=>v+'%'}, border:{color:'transparent'} }
        }
      }
    });
  } catch (err) { console.error('Timeline chart failed:', err); }
}

// ── Risk Zone Distribution Chart ───────────────────────────────────────────────
async function loadDistChart() {
  try {
    const data = await apiFetch('/api/risk-distribution');
    const ctx  = $('distChart');
    if (!ctx) return;
    if (distChart) distChart.destroy();

    const zones  = data.zones;  // [{start,end,color,label}, ...]
    // Colour each bar by zone
    const colors = data.labels.map((_, i) => {
      if (i < zones[1].start) return 'rgba(0,245,212,0.75)';   // ALLOW zone – cyan
      if (i < zones[2].start) return 'rgba(247,183,49,0.75)';  // OTP zone – amber
      return 'rgba(255,56,100,0.75)';                           // BLOCK zone – red
    });

    // Zone boundary plugin (vertical lines at threshold positions)
    const thresholdLines = {
      id: 'thresholdLines',
      afterDraw(chart) {
        const { ctx: c, chartArea: { top, bottom }, scales: { x } } = chart;
        // OTP threshold between index 4 and 5 (0-50% → 50-60%)
        // BLOCK threshold between index 7 and 8 (70-80% → 80-85%)
        [4.5, 7.5].forEach((idx, i) => {
          const xPx = x.getPixelForValue(idx);
          c.save();
          c.beginPath();
          c.moveTo(xPx, top);
          c.lineTo(xPx, bottom);
          c.strokeStyle = i === 0 ? 'rgba(247,183,49,0.6)' : 'rgba(255,56,100,0.6)';
          c.lineWidth = 2;
          c.setLineDash([6, 4]);
          c.stroke();
          c.restore();

          // Label
          c.save();
          c.fillStyle = i === 0 ? '#f7b731' : '#ff3864';
          c.font = 'bold 10px Inter,sans-serif';
          c.fillText(i === 0 ? '▲ OTP threshold 50%' : '▲ Block threshold 80%', xPx + 4, bottom - 6);
          c.restore();
        });
      }
    };

    distChart = new Chart(ctx, {
      type: 'bar',
      plugins: [thresholdLines],
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Transactions',
          data: data.counts,
          backgroundColor: colors,
          borderRadius: 4,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(12,18,33,0.95)',
            borderColor: 'rgba(0,245,212,0.25)',
            borderWidth: 1,
            titleColor: '#e2e8f0',
            bodyColor: '#94a3b8',
            padding: 10,
            callbacks: {
              title: ts => `Fraud Score: ${ts[0].label}`,
              label: c => {
                const idx = c.dataIndex;
                const zone = idx < zones[1].start ? '✅ ALLOW zone'
                           : idx < zones[2].start ? '⚠️ OTP zone'
                           : '🚫 BLOCK zone';
                return [` ${c.parsed.y.toLocaleString()} transactions`, ` Zone: ${zone}`];
              }
            }
          }
        },
        scales: {
          x: { grid:{display:false}, ticks:{color:'#64748b',font:{size:10}}, border:{color:'transparent'},
               title:{display:true, text:'Fraud Score (%) →  More Dangerous', color:'#4b5563', font:{size:11}} },
          y: { grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#64748b',font:{size:11}}, border:{color:'transparent'},
               title:{display:true, text:'Transactions', color:'#4b5563', font:{size:11}} }
        }
      }
    });

    // Update zone percentage counters
    const total = data.counts.reduce((a, b) => a + b, 0);
    const allowTotal = data.zone_totals.allow;
    const otpTotal   = data.zone_totals.otp;
    const blockTotal = data.zone_totals.block;
    const ap = $('zone-allow-pct'); if (ap) ap.textContent = ((allowTotal/total)*100).toFixed(1) + '%';
    const op = $('zone-otp-pct');   if (op) op.textContent = ((otpTotal/total)*100).toFixed(1) + '%';
    const bp = $('zone-block-pct'); if (bp) bp.textContent = ((blockTotal/total)*100).toFixed(1) + '%';

  } catch (err) { console.error('Dist chart failed:', err); }
}

// ── Prevention Form ────────────────────────────────────────────────────────────
async function handlePredict(e) {
  e.preventDefault();
  const btn    = $('predict-btn');
  const result = $('predict-result');

  const payload = {
    time_step:  parseFloat($('input-timestep').value)  || 25,
    in_degree:  parseFloat($('input-indegree').value)  || 3,
    out_degree: parseFloat($('input-outdegree').value) || 3,
    btc_vol_log:parseFloat($('input-vol').value)       || 5.0,
    is_exchange:parseFloat($('input-exchange').value)  || 0,
    is_service: parseFloat($('input-service').value)   || 0,
    suspicious: parseFloat($('input-suspicious').value)|| 0,
    local_edges:parseFloat($('input-edges').value)     || 10,
    lifetime:   parseFloat($('input-lifetime').value)  || 12,
  };

  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner"></span> Running Prevention Check…';

  try {
    const res  = await fetch(API + '/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    renderPreventionResult(data, payload);
    showToast('Prevention check complete', 'success');
    if (result) result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    showToast('Prevention check failed — check the API.', 'error');
    console.error(err);
  } finally {
    btn.disabled  = false;
    btn.innerHTML = '🛡️ Run Prevention Check';
  }
}

function renderPreventionResult(data, input) {
  const panel = $('predict-result');
  if (!panel) return;

  const prob     = data.fraud_probability;
  const pct      = (prob * 100).toFixed(1);
  const decision = data.decision;
  const cls      = { ALLOW: 'allow', OTP: 'otp', BLOCK: 'block' }[decision] || 'allow';
  const icon     = { ALLOW: '✅', OTP: '⚠️', BLOCK: '🚫' }[decision];

  // Fan-out ratio for display
  const fanRatio = input.out_degree > 0
    ? (input.out_degree / Math.max(input.in_degree, 1)).toFixed(1)
    : '—';

  panel.innerHTML = `
    <div class="result-header">
      <div class="result-icon ${cls}">${icon}</div>
      <div>
        <div class="result-action ${cls}">${data.action_text}</div>
        <div class="result-meta">
          Time Step T${input.time_step} &nbsp;·&nbsp;
          ${input.suspicious == 1 ? '⚠️ Flagged address detected &nbsp;·&nbsp;' : ''}
          ${input.is_exchange == 1 ? '🏦 Exchange wallet &nbsp;·&nbsp;' : ''}
          Wallet lifetime: ${input.lifetime} step(s) &nbsp;·&nbsp;
          Fan-out ratio: ${fanRatio}
        </div>
      </div>
    </div>

    <!-- Fraud probability bar with threshold markers -->
    <div style="position:relative;margin:20px 0 6px">
      <div class="fraud-bar-track">
        <div class="fraud-bar-fill" id="fraud-bar" style="width:0%"></div>
      </div>
      <!-- OTP threshold marker at 50% -->
      <div style="position:absolute;top:-6px;left:50%;transform:translateX(-50%);
                  font-size:9px;color:#f7b731;text-align:center;white-space:nowrap">
        ▼<br/>OTP<br/>50%
      </div>
      <!-- BLOCK threshold marker at 80% -->
      <div style="position:absolute;top:-6px;left:80%;transform:translateX(-50%);
                  font-size:9px;color:#ff3864;text-align:center;white-space:nowrap">
        ▼<br/>BLOCK<br/>80%
      </div>
    </div>
    <div class="fraud-bar-labels" style="margin-top:28px">
      <span style="color:var(--accent-cyan)">✅ ALLOW  (0–50%)</span>
      <span style="color:${data.color};font-weight:700">Fraud Score: ${pct}%</span>
      <span style="color:var(--accent-red)">🚫 BLOCK  (&gt;80%)</span>
    </div>

    <div class="result-metrics mt-24">
      <div class="metric-box">
        <div class="metric-box-label">Fraud Score</div>
        <div class="metric-box-value" style="color:${data.color}">${pct}%</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">Risk Level</div>
        <div class="metric-box-value" style="color:${data.color}">${data.risk_level}</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">Prevention Action</div>
        <div class="metric-box-value" style="font-size:15px;color:${data.color}">${decision}</div>
      </div>
    </div>

    <!-- Key graph signals -->
    <div class="metric-note mt-16" style="border-color:${data.color}33">
      <div class="note-icon">${icon}</div>
      <div>
        <strong>Key signals detected:</strong>
        ${input.suspicious == 1 ? ' 🚨 Transaction connects to previously flagged illicit address (+55% fraud score).' : ''}
        ${input.is_exchange == 1 ? ' 🏦 Exchange connection detected (−22% fraud score).' : ''}
        ${input.lifetime <= 2 ? ' ⚡ Very short-lived wallet (1–2 steps) — high suspicion (+18%).' : ''}
        ${(input.out_degree / Math.max(input.in_degree, 1)) > 5 ? ' 🌊 Extreme fan-out (&gt;5:1 out/in) — structuring pattern detected (+25%).' : ''}
        ${decision === 'ALLOW' ? ' ✅ No significant fraud signals. Transaction clears all prevention checks.' : ''}
      </div>
    </div>
  `;

  panel.classList.add('visible');
  setTimeout(() => {
    const bar = $('fraud-bar');
    if (bar) bar.style.width = pct + '%';
  }, 100);
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function initNav() {
  const links    = document.querySelectorAll('.nav-link[data-section]');
  const sections = document.querySelectorAll('.page-section');

  links.forEach(link => {
    link.addEventListener('click', () => {
      const target = link.dataset.section;
      links.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      sections.forEach(s => {
        s.style.display = (s.id === target + '-section') ? '' : 'none';
      });
    });
  });
}

// ── Chart.js global defaults ───────────────────────────────────────────────────
function setChartDefaults() {
  Chart.defaults.color       = '#64748b';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size   = 12;
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
async function init() {
  setChartDefaults();
  initNav();

  await Promise.all([
    checkHealth(),
    loadStats(),
    loadFeed(),
    loadTimelineChart(),
    loadDistChart(),
  ]);

  startFeedRefresh();

  const form = $('predict-form');
  if (form) form.addEventListener('submit', handlePredict);

  showToast('CryptoGuard prevention system online', 'success');
}

document.addEventListener('DOMContentLoaded', init);
