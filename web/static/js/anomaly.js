// 异常检测页 — v6.1（E_cosine + LOF）
let scoresChart = null;
let sensitivityChart = null;

async function init() {
    await Promise.all([loadSummary(), loadScores(), loadEvents(), loadSensitivity()]);
}

async function loadSummary() {
    const res = await fetch('/api/anomalies/summary');
    const data = await res.json();
    document.getElementById('stat-events').textContent = data.total_events ?? '-';
    document.getElementById('stat-verified').textContent = data.verified_events ?? '-';
    document.getElementById('stat-m4').textContent = data.total_m4_eq ?? '-';
    document.getElementById('stat-m5').textContent = data.total_m5_eq ?? '-';
    const desc = document.getElementById('model-desc');
    if (desc) {
        desc.textContent =
            `模型：${data.model || 'E_cosine'} ｜ 切分：${data.split || '65/35'} ｜ ` +
            `LOF 阈值 q=${data.lof_quantile ?? 0.98} ｜ 异常点 ${data.n_anomaly_points ?? '-'}`;
    }
}

async function loadScores() {
    const res = await fetch('/api/anomalies/scores?quantile=0.98');
    const data = await res.json();
    renderScoresChart(data);
}

function renderScoresChart(data) {
    if (!scoresChart) {
        scoresChart = echarts.init(document.getElementById('scores-chart'));
        window.addEventListener('resize', () => scoresChart.resize());
    }
    const dates = data.dates || [];
    const scores = data.scores || [];
    const thr = data.threshold ?? 0;
    const anomalyData = [];
    dates.forEach((d, i) => {
        if (scores[i] > 0 && scores[i] >= thr) anomalyData.push([d, scores[i]]);
    });

    scoresChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['LOF 得分', '异常点', '阈值'], top: 10 },
        grid: { left: 60, right: 30, top: 50, bottom: 50 },
        xAxis: {
            type: 'category', data: dates, boundaryGap: false,
            axisLabel: { formatter: v => (v ? String(v).substring(0, 10) : '') }
        },
        yAxis: { type: 'value', name: 'LOF 得分' },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
        series: [
            {
                name: 'LOF 得分', type: 'line', data: scores, symbol: 'none',
                lineStyle: { width: 1.2, color: '#5470c6' }, itemStyle: { color: '#5470c6' },
            },
            {
                name: '异常点', type: 'scatter', data: anomalyData, symbolSize: 10,
                itemStyle: { color: '#ee6666' },
            },
            {
                name: '阈值', type: 'line', data: dates.map(() => thr), symbol: 'none',
                lineStyle: { width: 1, color: '#fac858', type: 'dashed' },
                itemStyle: { color: '#fac858' },
            },
        ],
    }, true);
}

async function loadEvents() {
    const res = await fetch('/api/anomalies/events');
    const events = await res.json();
    const tbody = document.getElementById('events-body');
    tbody.innerHTML = (events || []).map(e => {
        const ok = e.correspondence === '✓';
        return `
            <tr>
                <td>${e.id}</td>
                <td>${e.start} ~ ${e.end}</td>
                <td>${e.n_points}</td>
                <td>${e.peak_score}</td>
                <td>${e.precip_mm} mm</td>
                <td><span class="tag tag-${e.attribution.includes('气象') ? 'warn' : 'pass'}">${e.attribution}</span></td>
                <td>${e.eq_m4}</td>
                <td>${e.eq_m5 > 0 ? `<strong>${e.eq_m5}</strong>` : '0'}</td>
                <td><span class="tag tag-${ok ? 'pass' : 'fail'}">${ok ? '✓ 有后续地震' : '✗ 无'}</span></td>
            </tr>`;
    }).join('');
    renderEarthquakeDetails(events || []);
}

function renderEarthquakeDetails(events) {
    const box = document.getElementById('earthquake-details');
    const withEq = events.filter(e => e.eq_m5 > 0 && e.eq_m5_details && e.eq_m5_details !== '无');
    if (!withEq.length) {
        box.innerHTML = '<p class="no-data">暂无 M5+ 对应</p>';
        return;
    }
    box.innerHTML = '<div class="eq-list">' + withEq.map(e => {
        const details = String(e.eq_m5_details).split(';').filter(s => s.trim());
        return `
            <div class="eq-item">
                <div class="eq-header">
                    <span class="eq-badge">事件 #${e.id}</span>
                    <span class="eq-date">${e.start} ~ ${e.end}</span>
                    <span class="eq-count">${e.eq_m5} 个 M5+</span>
                </div>
                <ul class="eq-details">
                    ${details.map(d => `<li>${d.trim()}</li>`).join('')}
                </ul>
            </div>`;
    }).join('') + '</div>';
}

async function loadSensitivity() {
    const res = await fetch('/api/anomalies/sensitivity');
    const data = await res.json();
    if (!sensitivityChart) {
        sensitivityChart = echarts.init(document.getElementById('sensitivity-chart'));
        window.addEventListener('resize', () => sensitivityChart.resize());
    }
    sensitivityChart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 30, top: 30, bottom: 40 },
        xAxis: { type: 'category', data: data.map(d => d.quantile) },
        yAxis: { type: 'value', name: '异常点数' },
        series: [{
            name: '异常点数', type: 'bar', data: data.map(d => d.n_points ?? d.n_anomalies),
            itemStyle: { color: '#ff7f0e' },
            label: { show: true, position: 'top' },
        }],
    }, true);
}

init();
