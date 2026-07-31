let weeklyChartObj = null;
let expenseStatsChartObj = null;
let balanceHistoryChartObj = null;
let currentThreshold = 85;

let liveTotalAnalyzed = 10000;
let liveTotalSavings = 125038.7;

let alertsQueue = [
    {
        trans_num: "TRX-FRAUD-8842",
        amt_mad: 4850.0,
        category: "shopping_net (Shopping Web)",
        trans_date_trans_time: "Aujourd'hui à 14:22",
        risk_score: 98.7,
        verdict: "ALERTE FRAUDE BLOQUÉE",
        badge_color: "danger"
    }
];
let currentInspectedAlert = null;

document.addEventListener('DOMContentLoaded', () => {
    initBankdash();

    const form = document.getElementById('transactionForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await analyzeTransaction();
        });
    }
});

function incrementLiveCounter(isFraud, amtMad) {
    liveTotalAnalyzed += 1;
    const totalEl = document.getElementById('kpi-total');
    if (totalEl) totalEl.textContent = liveTotalAnalyzed.toLocaleString('fr-FR');

    if (isFraud && amtMad) {
        liveTotalSavings += amtMad;
        const savingsEl = document.getElementById('kpi-savings');
        if (savingsEl) savingsEl.textContent = `${liveTotalSavings.toLocaleString('fr-FR', {maximumFractionDigits: 1})} MAD`;
    }
}

function onThresholdChange(val) {
    currentThreshold = parseInt(val);
    const badge = document.getElementById('thresholdValBadge');
    const desc = document.getElementById('thresholdDescText');
    if (badge) badge.textContent = `${currentThreshold}%`;
    
    if (desc) {
        if (currentThreshold < 70) {
            desc.innerHTML = `<strong>Sensibilité Élevée :</strong> Capture maximale des fraudes, hausse des alertes de précaution.`;
        } else if (currentThreshold <= 90) {
            desc.innerHTML = `<strong>Matrice de Coût Optimisée (${currentThreshold}%) :</strong> Équilibre idéal entre capture et faux positifs.`;
        } else {
            desc.innerHTML = `<strong>Seuil Strict (${currentThreshold}%) :</strong> Blocage sur certitude quasi-absolue. Risque de rater des micro-fraudes.`;
        }
    }
}

function injectDemoFraud() {
    const randomId = Math.floor(1000 + Math.random() * 9000);
    const categories = ["shopping_net (Shopping Web)", "grocery_pos (Supermarché)", "travel (Voyages & Air)", "misc_net (Paiement ATM)"];
    const chosenCat = categories[Math.floor(Math.random() * categories.length)];
    const score = (92.0 + Math.random() * 7.5).toFixed(1);
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const demoFraud = {
        trans_num: `TRX-FRAUD-${randomId}`,
        amt_mad: Math.floor(1800 + Math.random() * 7500),
        category: chosenCat,
        trans_date_trans_time: `Aujourd'hui à ${timeStr}`,
        risk_score: parseFloat(score),
        verdict: "ALERTE FRAUDE BLOQUÉE",
        badge_color: "danger"
    };
    
    alertsQueue.unshift(demoFraud);
    renderAlertsQueue();
    incrementLiveCounter(true, demoFraud.amt_mad);
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));

    const activeNav = document.querySelector(`.nav-item[href="#${tabId}"]`);
    if (activeNav) activeNav.classList.add('active');

    const activeTab = document.getElementById(`tab-${tabId}`);
    if (activeTab) activeTab.classList.add('active');

    const pageTitle = document.getElementById('pageTitle');
    if (tabId === 'dashboard') pageTitle.textContent = 'Overview';
    else if (tabId === 'tester') pageTitle.textContent = 'Transactions & Test';
    else if (tabId === 'analytics') pageTitle.textContent = 'Analytiques Risque';
    else if (tabId === 'history') {
        pageTitle.textContent = 'Historique & Audit des Alertes';
        loadAuditHistory();
    }
}

async function initBankdash() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (data.status === 'success') {
            liveTotalAnalyzed = data.total_transactions || 10000;
            liveTotalSavings = data.total_at_risk_mad || 125038.7;
            
            document.getElementById('kpi-total').textContent = liveTotalAnalyzed.toLocaleString('fr-FR');
            
            // Metrique Précision des Alertes (Precision)
            const precVal = data.precision || 95.2;
            document.getElementById('kpi-pct').textContent = `${precVal}%`;
            const kpiSub = document.getElementById('kpi-subtext');
            if (kpiSub && data.precision_str) {
                kpiSub.textContent = data.precision_str;
            }
            
            document.getElementById('kpi-savings').textContent = `${liveTotalSavings.toLocaleString('fr-FR', {maximumFractionDigits: 1})} MAD`;

            renderWeeklyChart(data.hourly_labels, data.hourly_fraud_rates);
            renderExpenseChart(data.blocked_categories);
            renderBalanceHistoryChart(data.hourly_labels, data.hourly_fraud_rates);
            loadRecentTransactionsList();
            renderAlertsQueue();
            loadAuditHistory();
        }
    } catch (err) {
        console.error('Erreur lors du chargement des statistiques Bankdash :', err);
    }
}

let streamInterval = null;

function toggleLiveStream() {
    const btn = document.getElementById('liveStreamBtn');
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
        if (btn) {
            btn.textContent = 'Activer le Flux Temps Réel';
            btn.className = 'btn btn-indigo';
        }
    } else {
        if (btn) {
            btn.textContent = 'Flux Actif (Pause)';
            btn.className = 'btn btn-secondary';
        }
        streamOneTransaction();
        streamInterval = setInterval(streamOneTransaction, 3000);
    }
}

async function streamOneTransaction() {
    const radarContainer = document.getElementById('recentTransList');
    if (!radarContainer) return;

    try {
        const response = await fetch(`/api/stream_transaction?threshold=${currentThreshold}`);
        const data = await response.json();

        if (data.status === 'success') {
            const isFraud = data.badge_color === 'danger' || data.risk_score >= currentThreshold;

            incrementLiveCounter(isFraud, data.amt_mad);

            if (isFraud) {
                // ZONE 2 : LA FILE D'ATTENTE DES FRAUDES (À TRAITER)
                if (!alertsQueue.some(a => a.trans_num === data.trans_num)) {
                    alertsQueue.unshift(data);
                    renderAlertsQueue();
                }
            } else {
                // ZONE 1 : LE RADAR (FLUX LÉGITIME EN DIRECT)
                const newItemHtml = `
                    <div class="trans-item animated-item" style="background-color: rgba(22, 219, 204, 0.05); border-left: 4px solid #10b981;">
                        <div class="trans-left">
                            <div class="trans-icon-circle" style="background-color: rgba(22, 219, 204, 0.15); color: #0d9488;">
                                L
                            </div>
                            <div>
                                <div class="trans-name">${data.category} (LÉGITIME)</div>
                                <div class="trans-date">${data.trans_date_trans_time} - Risk: ${data.risk_score}%</div>
                            </div>
                        </div>
                        <div class="trans-amt amt-green">
                            +${data.amt_mad.toLocaleString()} MAD
                        </div>
                    </div>
                `;
                radarContainer.insertAdjacentHTML('afterbegin', newItemHtml);

                while (radarContainer.children.length > 3) {
                    radarContainer.removeChild(radarContainer.lastElementChild);
                }
            }
        }
    } catch (err) {
        console.error('Erreur flux temps réel :', err);
    }
}

function renderAlertsQueue() {
    const queueContainer = document.getElementById('alertsQueueList');
    const badgePill = document.getElementById('alertQueueBadge');
    if (!queueContainer) return;

    if (badgePill) badgePill.textContent = `${alertsQueue.length} alerte${alertsQueue.length > 1 ? 's' : ''}`;

    if (alertsQueue.length === 0) {
        queueContainer.innerHTML = `
            <div class="empty-queue-msg" id="emptyQueueMsg">
                Aucune alerte en attente. Le système écoute le réseau.
            </div>
        `;
        return;
    }

    queueContainer.innerHTML = alertsQueue.map(alert => `
        <div class="alert-card-zone2 animated-item">
            <div class="alert-card-header">
                <span class="alert-trans-id">DOSSIER ${alert.trans_num}</span>
                <span class="alert-score-badge">Score : ${alert.risk_score}%</span>
            </div>
            <div class="alert-card-body">
                <div class="alert-info-col">
                    <div class="alert-cat">${alert.category}</div>
                    <div class="alert-date">${alert.trans_date_trans_time}</div>
                </div>
                <div class="alert-amt-red">-${alert.amt_mad.toLocaleString('fr-FR', {minimumFractionDigits: 2})} MAD</div>
            </div>
            <button class="btn-analyze-alert" onclick="inspectAlert('${alert.trans_num}')">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                Traiter l'Alerte (SHAP)
            </button>
        </div>
    `).join('');
}

async function inspectAlert(transNum) {
    const alertData = alertsQueue.find(a => a.trans_num === transNum);
    if (!alertData) return;

    currentInspectedAlert = alertData;

    try {
        const res = await fetch('/api/explain_transaction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(alertData)
        });
        const exp = await res.json();

        if (exp.status === 'success') {
            document.getElementById('shapCaseTitle').textContent = `Dossier Alerte : ${exp.trans_num} (Score : ${exp.risk_score}%)`;
            document.getElementById('shapExplanationText').textContent = exp.explanation_text;

            const factorsList = document.getElementById('shapFactorsList');
            factorsList.innerHTML = exp.factors.map(f => `
                <div class="factor-item">
                    <div>
                        <div class="factor-name">${f.feature}</div>
                        <div style="font-size:0.75rem; color:#64748b;">${f.description}</div>
                    </div>
                    <div class="factor-impact">${f.impact}</div>
                </div>
            `).join('');

            document.getElementById('shapInspectionCard').classList.remove('hidden');
        }
    } catch (e) {
        console.error("Erreur inspection SHAP :", e);
    }
}

function closeShapPanel() {
    const card = document.getElementById('shapInspectionCard');
    if (card) card.classList.add('hidden');
    currentInspectedAlert = null;
}

async function resolveCurrentAlert(actionType) {
    if (!currentInspectedAlert) return;

    try {
        const res = await fetch('/api/alerts/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trans_num: currentInspectedAlert.trans_num,
                action: actionType,
                amt_mad: currentInspectedAlert.amt_mad,
                category: currentInspectedAlert.category,
                trans_date_trans_time: currentInspectedAlert.trans_date_trans_time,
                risk_score: currentInspectedAlert.risk_score
            })
        });
        const result = await res.json();

        if (result.status === 'success') {
            alertsQueue = alertsQueue.filter(a => a.trans_num !== currentInspectedAlert.trans_num);
            renderAlertsQueue();
            closeShapPanel();
            loadAuditHistory();
        }
    } catch (e) {
        console.error("Erreur résolution alerte :", e);
    }
}

async function loadAuditHistory(showFeedback = false) {
    const tableBody = document.getElementById('auditTableBody');
    const btn = document.getElementById('btnRefreshAudit');
    const icon = document.getElementById('refreshAuditIcon');
    const text = document.getElementById('refreshAuditText');
    const viewModeSelect = document.getElementById('auditViewMode');
    
    if (!tableBody) return;

    if (btn && showFeedback) {
        btn.disabled = true;
        if (icon) icon.style.transform = 'rotate(360deg)';
        if (text) text.textContent = 'Actualisation...';
    }

    try {
        const res = await fetch(`/api/audit/logs?_t=${Date.now()}`);
        const data = await res.json();

        if (data.status === 'success') {
            const logs = data.logs || [];
            const mode = viewModeSelect ? viewModeSelect.value : 'synthetic';
            
            let displayLogs = [];
            
            if (mode === 'synthetic') {
                // Consolidations par transaction (1ère occurrence = l'état le plus récent)
                const seenTrans = new Set();
                for (const l of logs) {
                    if (!seenTrans.has(l.trans_num)) {
                        seenTrans.add(l.trans_num);
                        displayLogs.push(l);
                    }
                }
            } else {
                // Mode brut : tous les événements consignés dans le CSV
                displayLogs = logs;
            }

            let fpCount = 0;
            let blockCount = 0;

            logs.forEach(l => {
                if (l.action === 'false_positive') fpCount++;
                else if (l.action === 'block_card' || l.action === 'send_otp') blockCount++;
            });

            const totalEl = document.getElementById('auditTotalLogged');
            const fpEl = document.getElementById('auditFpCount');
            const blockEl = document.getElementById('auditBlockCount');

            if (totalEl) totalEl.textContent = logs.length;
            if (fpEl) fpEl.textContent = fpCount;
            if (blockEl) blockEl.textContent = blockCount;

            if (displayLogs.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; padding: 24px; color: #718ebf;">
                            Aucune action d'audit consignée pour le moment. Traitez des alertes depuis le Dashboard.
                        </td>
                    </tr>
                `;
            } else {
                // Ensemble des transactions annulées pour bloquer les boutons obsolètes même en mode brut
                const cancelledTransNums = new Set(logs.filter(l => l.action === 'undo_action').map(l => l.trans_num));

                tableBody.innerHTML = displayLogs.map(l => {
                    let badgeClass = 'status-badge-block';
                    let labelText = l.action_label || 'ACTION ENREGISTRÉE';

                    if (l.action === 'false_positive' || labelText.includes('FAUX POSITIF')) {
                        badgeClass = 'status-badge-fp';
                    } else if (l.action === 'send_otp' || labelText.includes('OTP')) {
                        badgeClass = 'status-badge-otp';
                    } else if (l.action === 'undo_action' || labelText.includes('ANNULÉ')) {
                        badgeClass = 'status-badge-undo';
                        labelText = 'ANNULÉ - RETOUR FILE';
                    }

                    const isCancelled = l.action === 'undo_action' || cancelledTransNums.has(l.trans_num);
                    const isUndoable = !isCancelled;
                    const amtFormatted = l.amt_mad ? parseFloat(l.amt_mad).toLocaleString('fr-FR', {minimumFractionDigits: 2}) + ' MAD' : '0.00 MAD';
                    const analystName = l.analyst || 'Analyste Risque';

                    return `
                        <tr>
                            <td style="font-weight: 500; font-size: 0.82rem; color: #343c6a;">${l.timestamp || '-'}</td>
                            <td style="font-weight: 700; color: #1814f3;">${l.trans_num}</td>
                            <td>
                                <span style="font-weight: 600;">${analystName}</span>
                            </td>
                            <td>${l.category || 'Achat Net'}</td>
                            <td style="font-weight: 700; color: #343c6a;">${amtFormatted}</td>
                            <td>
                                <span class="audit-status-pill ${badgeClass}">
                                    ${labelText}
                                </span>
                            </td>
                            <td>
                                ${isUndoable ? `
                                    <button class="btn-undo-action" onclick="undoAlertAction('${l.trans_num}', ${l.amt_mad || 0}, '${l.category || ''}', '${l.trans_date || ''}')" title="Restaurer l'alerte dans la file d'attente (Zone 2)">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                                        Annuler l'action
                                    </button>
                                ` : `
                                    <span style="font-size: 0.78rem; color: #64748b; font-style: italic; font-weight: 500;">
                                        ✓ Annulé (Restauré en Zone 2)
                                    </span>
                                `}
                            </td>
                        </tr>
                    `;
                }).join('');
            }
        }
    } catch (e) {
        console.error("Erreur chargement logs audit :", e);
    } finally {
        if (btn && showFeedback) {
            setTimeout(() => {
                btn.disabled = false;
                if (icon) icon.style.transform = 'none';
                if (text) text.textContent = 'Journal Actualisé !';
                setTimeout(() => {
                    if (text) text.textContent = 'Actualiser le Journal';
                }, 1500);
            }, 300);
        }
    }
}

async function undoAlertAction(transNum, amtMad, category, transDate) {
    if (!confirm(`Voulez-vous annuler l'action sur la transaction ${transNum} et la replacer dans la file d'attente ?`)) {
        return;
    }

    try {
        const res = await fetch('/api/alerts/undo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trans_num: transNum,
                amt_mad: amtMad,
                category: category,
                trans_date_trans_time: transDate
            })
        });

        const result = await res.json();

        if (result.status === 'success') {
            if (result.restored_alert && !alertsQueue.some(a => a.trans_num === transNum)) {
                alertsQueue.unshift(result.restored_alert);
                renderAlertsQueue();
            }

            await loadAuditHistory();
        }
    } catch (e) {
        console.error("Erreur annulation alerte :", e);
    }
}

async function loadRecentTransactionsList() {
    const listContainer = document.getElementById('recentTransList');
    if (!listContainer) return;

    listContainer.innerHTML = '';

    for (let i = 0; i < 3; i++) {
        try {
            const res = await fetch(`/api/stream_transaction?threshold=${currentThreshold}`);
            const item = await res.json();
            if (item.status === 'success') {
                const isFraud = item.badge_color === 'danger';
                const badgeBg = isFraud ? 'rgba(254, 92, 115, 0.15)' : 'rgba(22, 219, 204, 0.15)';
                const badgeColor = isFraud ? '#fe5c73' : '#0d9488';
                const amtClass = isFraud ? 'amt-red' : 'amt-green';
                const amtSign = isFraud ? '-' : '+';
                const iconChar = isFraud ? 'F' : 'L';

                const html = `
                    <div class="trans-item">
                        <div class="trans-left">
                            <div class="trans-icon-circle" style="background-color: ${badgeBg}; color: ${badgeColor};">
                                ${iconChar}
                            </div>
                            <div>
                                <div class="trans-name">${item.category} (${item.verdict})</div>
                                <div class="trans-date">${item.trans_date_trans_time}</div>
                            </div>
                        </div>
                        <div class="trans-amt ${amtClass}">
                            ${amtSign}${item.amt_mad.toLocaleString()} MAD
                        </div>
                    </div>
                `;
                listContainer.innerHTML += html;
            }
        } catch (e) {
            console.error('Erreur chargement initial :', e);
        }
    }
}

function renderWeeklyChart(labels, values) {
    const ctx = document.getElementById('weeklyActivityChart');
    if (!ctx) return;

    if (weeklyChartObj) weeklyChartObj.destroy();

    weeklyChartObj = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Taux de Fraude (%)',
                data: values,
                backgroundColor: '#1814f3',
                borderRadius: 8,
                barThickness: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#718ebf' } },
                y: { grid: { color: '#e6eff5' }, ticks: { color: '#718ebf' } }
            }
        }
    });
}

function renderExpenseChart(blockedCategories) {
    const ctx = document.getElementById('expenseStatsChart');
    if (!ctx) return;

    if (expenseStatsChartObj) expenseStatsChartObj.destroy();

    const defaultCat = {
        'Shopping & Web': 45,
        'Supermarchés': 25,
        'Voyage & Transport': 15,
        'Distributeur / ATM': 10,
        'Divertissement': 5
    };

    const dataMap = blockedCategories || defaultCat;
    const labels = Object.keys(dataMap);
    const values = Object.values(dataMap);
    const colors = ['#fe5c73', '#ff82ac', '#1814f3', '#16dbcc', '#ffbb38'];

    expenseStatsChartObj = new Chart(ctx.getContext('2d'), {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#343c6a', font: { family: 'Inter', size: 11, weight: '600' } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` ${context.label}: ${context.raw} fraudes bloquées`;
                        }
                    }
                }
            }
        }
    });
}

function renderBalanceHistoryChart(labels, values) {
    const ctx = document.getElementById('balanceHistoryChart');
    if (!ctx) return;

    if (balanceHistoryChartObj) balanceHistoryChartObj.destroy();

    balanceHistoryChartObj = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Taux de Fraude par Heure (%)',
                data: values,
                borderColor: '#1814f3',
                backgroundColor: 'rgba(24, 20, 243, 0.08)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#1814f3'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#e6eff5' }, ticks: { color: '#718ebf' } },
                y: { grid: { color: '#e6eff5' }, ticks: { color: '#718ebf' } }
            }
        }
    });
}

async function loadSample(sampleType) {
    try {
        const response = await fetch(`/api/sample?type=${sampleType}`);
        const data = await response.json();

        if (data.status === 'success') {
            document.getElementById('amt').value = data.amt;
            document.getElementById('category').value = data.category;
            document.getElementById('gender').value = data.gender;
            if (data.dob) document.getElementById('dob').value = data.dob.split(' ')[0];
            document.getElementById('lat').value = data.lat;
            document.getElementById('long').value = data.long;
            document.getElementById('merch_lat').value = data.merch_lat;
            document.getElementById('merch_long').value = data.merch_long;
            
            const nowIso = data.trans_date_trans_time ? data.trans_date_trans_time.replace(' ', 'T') : '';
            if (nowIso) document.getElementById('trans_date_trans_time').value = nowIso;

            await analyzeTransaction();
        }
    } catch (err) {
        console.error('Erreur lors du chargement de l\'échantillon :', err);
    }
}

async function analyzeTransaction() {
    const amt = parseFloat(document.getElementById('amt').value);
    const category = document.getElementById('category').value;
    const gender = document.getElementById('gender').value;
    const dob = document.getElementById('dob').value;
    const lat = parseFloat(document.getElementById('lat').value);
    const long = parseFloat(document.getElementById('long').value);
    const merch_lat = parseFloat(document.getElementById('merch_lat').value);
    const merch_long = parseFloat(document.getElementById('merch_long').value);
    const timeVal = document.getElementById('trans_date_trans_time').value;

    const payload = {
        amt, category, gender, dob, lat, long, merch_lat, merch_long,
        trans_date_trans_time: timeVal ? timeVal.replace('T', ' ') : '2026-07-21 12:00:00'
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.status === 'success') {
            displayDecision(result);
        } else {
            alert(`Erreur : ${result.message}`);
        }
    } catch (err) {
        console.error('Erreur d\'analyse :', err);
    }
}

function displayDecision(res) {
    document.getElementById('decisionPlaceholder').classList.add('hidden');
    document.getElementById('decisionResult').classList.remove('hidden');

    const banner = document.getElementById('verdictBanner');
    banner.className = 'verdict-banner';
    banner.textContent = res.verdict;

    const fill = document.getElementById('gaugeFill');
    const val = document.getElementById('riskScoreVal');
    val.textContent = `${res.risk_score}%`;
    fill.style.width = `${res.risk_score}%`;

    if (res.badge_color === 'danger') {
        banner.classList.add('banner-red');
        fill.style.backgroundColor = '#fe5c73';
        val.style.color = '#fe5c73';
    } else if (res.badge_color === 'warning') {
        banner.classList.add('banner-yellow');
        fill.style.backgroundColor = '#ffbb38';
        val.style.color = '#d97706';
    } else {
        banner.classList.add('banner-green');
        fill.style.backgroundColor = '#16dbcc';
        val.style.color = '#0d9488';
    }

    document.getElementById('valAmtMad').textContent = `${res.amt_mad.toLocaleString()} MAD`;
    document.getElementById('valDistKm').textContent = `${res.distance_km} km`;
    document.getElementById('valThreshold').textContent = `${(res.optimal_threshold * 100).toFixed(1)}%`;
}
