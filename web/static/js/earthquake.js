// 震例对应验证页
const API = '';

async function init() {
    const [summary, verif, events, sens, timeline] = await Promise.all([
        fetch(`${API}/api/earthquakes/summary`).then(r => r.json()),
        fetch(`${API}/api/earthquakes/verification`).then(r => r.json()),
        fetch(`${API}/api/earthquakes/events`).then(r => r.json()),
        fetch(`${API}/api/earthquakes/sensitivity`).then(r => r.json()),
        fetch(`${API}/api/earthquakes/timeline`).then(r => r.json()),
    ]);

    document.getElementById('stat-key').textContent = summary.n_key_eqs;
    document.getElementById('stat-test').textContent = summary.n_in_test;
    document.getElementById('stat-matched').textContent = summary.n_matched;
    document.getElementById('stat-events').textContent = summary.n_events;
    document.getElementById('stat-m5').textContent = summary.events_with_m5;
    document.getElementById('model-desc').textContent =
        `模型：${summary.model} ｜ 切分：${summary.split} ｜ LOF 阈值 q=${summary.lof_quantile}`;

    renderVerifTable(verif);
    renderEventsTable(events);
    renderTimeline(timeline);
    renderSens(sens);
}

function resultBadge(result) {
    if (result.includes('✓')) return `<span class="badge badge-ok">${result}</span>`;
    if (result.includes('✗')) return `<span class="badge badge-fail">${result}</span>`;
    return `<span class="badge badge-muted">${result}</span>`;
}

function renderVerifTable(rows) {
    const tbody = document.getElementById('verif-body');
    tbody.innerHTML = rows.map(r => `
        <tr>
            <td>${r.date}</td>
            <td>${r.place}</td>
            <td><strong>${r.mag}</strong></td>
            <td>${r.in_test ? '是' : '否'}</td>
            <td>${r.n_events}</td>
            <td>${r.days_before ?? '—'}</td>
            <td>${r.weather}</td>
            <td>${resultBadge(r.result)}</td>
            <td style="font-size:12px;color:#666">${r.detail}</td>
        </tr>
    `).join('');
}

function renderEventsTable(rows) {
    const tbody = document.getElementById('events-body');
    tbody.innerHTML = rows.map(e => {
        const eqs = (e.subsequent || []).slice(0, 3).map(s =>
            `${s.date} M${s.mag}`
        ).join('；') + (e.subsequent && e.subsequent.length > 3 ? ' …' : '');
        return `
        <tr>
            <td>${e.id}</td>
            <td>${e.start} ~ ${e.end}</td>
            <td>${e.n_points}</td>
            <td>${e.peak_score}</td>
            <td>${e.n_m45}</td>
            <td>${e.n_m5 > 0 ? `<strong>${e.n_m5}</strong>` : '0'}</td>
            <td style="font-size:12px">${eqs || '—'}</td>
        </tr>`;
    }).join('');
}

function renderTimeline(data) {
    const chart = echarts.init(document.getElementById('timeline-chart'));
    const events = data.events || [];
    const keyEqs = data.key_eqs || [];

    const eventMarks = events.map(e => ({
        name: `事件${e.id}`,
        xAxis: e.start,
        itemStyle: { color: '#d62728' },
        label: { formatter: `E${e.id}`, position: 'insideEndTop', fontSize: 10 }
    }));

    const eqMarks = keyEqs.map(eq => ({
        name: eq.mag,
        xAxis: eq.date,
        lineStyle: {
            color: eq.in_test ? (eq.result.includes('✓') ? '#2ca02c' : '#d62728') : '#999',
            type: eq.in_test ? 'solid' : 'dashed',
            width: 1.5
        },
        label: {
            formatter: `${eq.mag}${eq.in_test ? '' : ' (训)'}`,
            position: 'insideEndTop',
            fontSize: 10,
            color: eq.in_test ? (eq.result.includes('✓') ? '#2ca02c' : '#d62728') : '#999'
        }
    }));

    chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['残差', 'LOF事件', '关键地震'], top: 10 },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: data.dates,
            axisLabel: { formatter: v => v.substring(0, 7) }
        },
        yAxis: { type: 'value', name: '残差' },
        series: [
            {
                name: '残差', type: 'line', data: data.residual,
                lineStyle: { width: 0.8, color: '#666' },
                itemStyle: { color: '#666' }, showSymbol: false,
                markLine: {
                    symbol: 'none',
                    data: [{ yAxis: 0, lineStyle: { color: '#333', width: 0.5 } }]
                        .concat(eqMarks)
                },
                markPoint: {
                    symbol: 'pin', symbolSize: 28,
                    data: eventMarks,
                    itemStyle: { color: '#d62728' }
                }
            },
            {
                name: 'LOF事件', type: 'scatter', data: [],
                itemStyle: { color: '#d62728' }
            },
            {
                name: '关键地震', type: 'scatter', data: [],
                itemStyle: { color: '#1f77b4' }
            }
        ]
    });
    window.addEventListener('resize', () => chart.resize());
}

function renderSens(rows) {
    const chart = echarts.init(document.getElementById('sens-chart'));
    chart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 40, top: 40, bottom: 40 },
        xAxis: {
            type: 'category',
            name: '分位数',
            data: rows.map(r => String(r.quantile))
        },
        yAxis: [
            { type: 'value', name: '异常点数' },
            { type: 'value', name: '阈值', min: 'dataMin' }
        ],
        series: [
            {
                name: '异常点数', type: 'bar', data: rows.map(r => r.n_points),
                itemStyle: { color: '#ff7f0e' }, barWidth: 28
            },
            {
                name: '阈值', type: 'line', yAxisIndex: 1,
                data: rows.map(r => r.threshold),
                itemStyle: { color: '#1f77b4' },
                lineStyle: { width: 2 }
            }
        ]
    });
    window.addEventListener('resize', () => chart.resize());
}

init();
