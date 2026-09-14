// 模型预测页 — v6.1 主结果 + v6.0 多变量对照
let predChart = null;
let residualChart = null;
let yearChart = null;
let mode = 'v61';
let currentWindow = 60;
let currentVar = 'radon';

const COLORS = {
    naive: '#999999', rf: '#5470c6', lstm: '#91cc75',
    lstm_rf: '#fac858', cnn_lstm: '#ee6666', e_cosine: '#7b61ff',
};
const NAMES = {
    naive: 'Naive', rf: '随机森林', lstm: 'LSTM',
    lstm_rf: 'LSTM-RF', cnn_lstm: 'CNN-LSTM', e_cosine: 'E_cosine ★',
};
const YEARS = ['2022', '2023', '2024', '2025', '2026'];

async function init() {
    const modeSelect = document.getElementById('modeSelect');
    modeSelect.addEventListener('change', e => {
        mode = e.target.value;
        toggleModeUI();
        refresh();
    });
    document.getElementById('varSelect').addEventListener('change', e => {
        currentVar = e.target.value;
        if (mode === 'multivar') loadMultivar();
    });
    document.getElementById('windowSelect').addEventListener('change', e => {
        currentWindow = parseInt(e.target.value, 10);
        if (mode === 'multivar') loadMultivar();
    });
    document.querySelectorAll('#model-checkboxes input').forEach(cb => {
        cb.addEventListener('change', () => { if (mode === 'v61') loadV61(); });
    });
    document.querySelectorAll('#mv-model-checkboxes input').forEach(cb => {
        cb.addEventListener('change', () => { if (mode === 'multivar') loadMultivar(); });
    });
    toggleModeUI();
    await refresh();
}

function toggleModeUI() {
    const isV61 = mode === 'v61';
    document.getElementById('multivar-controls').style.display = isV61 ? 'none' : '';
    document.getElementById('multivar-window').style.display = isV61 ? 'none' : '';
    document.getElementById('v61-models').style.display = isV61 ? '' : 'none';
    document.getElementById('multivar-models').style.display = isV61 ? 'none' : '';
    document.getElementById('year-r2-section').style.display = isV61 ? '' : 'none';
    document.getElementById('mode-desc').textContent = isV61
        ? 'v6.1：训练 2015~2022-05-15，测试 2022-05-16~2026-05。默认只显示 E_cosine ★，勾选其他模型可对比。滚轮缩放后线条会自动加粗；工具栏可框选缩放/重置。'
        : 'v6.0 多变量对照（80/20 切分，测试自 2024-01）。结论：水氡单变量最优，加入水位/气象反而降低精度。';
}

async function refresh() {
    if (mode === 'v61') await loadV61();
    else await loadMultivar();
}

async function loadV61() {
    const checked = Array.from(document.querySelectorAll('#model-checkboxes input:checked'))
        .map(cb => cb.value);
    if (!checked.length) return;

    const res = await fetch('/api/predictions/all');
    const data = await res.json();
    const chartData = {
        dates: data.dates,
        y_true: data.y_true,
        models: {},
    };
    checked.forEach(name => {
        if (data.models[name]) {
            chartData.models[name] = {
                display: data.models[name].display || NAMES[name],
                color: data.models[name].color || COLORS[name],
                y_pred: data.models[name].y_pred,
            };
        }
    });
    renderPredictionChart(chartData, checked);
    renderResidualChart(chartData, checked);
    await loadV61Metrics(checked);
    await loadYearR2();
}

async function loadV61Metrics(checked) {
    const res = await fetch('/api/metrics');
    const payload = await res.json();
    const models = payload.models || payload; // 兼容
    const tbody = document.getElementById('metrics-body');
    tbody.innerHTML = checked.map(name => {
        const m = models[name];
        if (!m) return '';
        return `<tr>
            <td><span style="color:${COLORS[name]}">●</span> ${m.display || NAMES[name]}</td>
            <td>${(m.RMSE ?? 0).toFixed(4)}</td>
            <td>${(m.MAE ?? 0).toFixed(4)}</td>
            <td>${m.MAPE === null || m.MAPE === undefined ? '—' : Number(m.MAPE).toFixed(2)}</td>
            <td><strong>${(m.R2 ?? 0).toFixed(4)}</strong></td>
        </tr>`;
    }).join('');
}

async function loadYearR2() {
    const res = await fetch('/api/year_r2');
    const yearR2 = await res.json();
    const rows = ['e_cosine', 'cnn_lstm', 'lstm_rf', 'rf', 'naive']
        .filter(k => yearR2[k]);
    const tbody = document.getElementById('year-r2-body');
    tbody.innerHTML = rows.map(k => {
        const y = yearR2[k];
        return `<tr>
            <td><span style="color:${COLORS[k]}">●</span> ${NAMES[k]}</td>
            ${YEARS.map(yy => {
                const v = y[yy];
                if (v === undefined || v === null) return '<td>—</td>';
                const cls = v < 0 ? 'color:#ff4d4f' : (v > 0.8 ? 'color:#52c41a' : '');
                return `<td style="${cls}">${Number(v).toFixed(3)}</td>`;
            }).join('')}
        </tr>`;
    }).join('');

    if (!yearChart) {
        yearChart = echarts.init(document.getElementById('year-r2-chart'));
        window.addEventListener('resize', () => yearChart.resize());
    }
    yearChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { top: 10 },
        grid: { left: 50, right: 20, top: 40, bottom: 30 },
        xAxis: { type: 'category', data: YEARS },
        yAxis: { type: 'value', name: 'R²' },
        series: rows.map(k => ({
            name: NAMES[k],
            type: 'line',
            data: YEARS.map(yy => yearR2[k][yy] ?? null),
            itemStyle: { color: COLORS[k] },
            lineStyle: { color: COLORS[k], width: 2 },
            symbol: 'circle',
            symbolSize: 6,
        })),
    }, true);
}

async function loadMultivar() {
    const checked = Array.from(document.querySelectorAll('#mv-model-checkboxes input:checked'))
        .map(cb => cb.value);
    if (!checked.length) return;

    const res = await fetch(`/api/predictions/${currentVar}/${currentWindow}?model=cnn_lstm`);
    const data = await res.json();
    if (data.error) {
        console.warn(data.error);
        return;
    }
    // 逐模型拉取
    const models = {};
    await Promise.all(checked.map(async name => {
        const r = await fetch(`/api/predictions/${currentVar}/${currentWindow}?model=${name}`);
        const d = await r.json();
        if (!d.error) {
            models[name] = {
                display: NAMES[name] || name,
                color: COLORS[name] || '#999',
                y_pred: d.y_pred,
                metrics: d.metrics,
            };
        }
    }));
    const chartData = { dates: data.dates, y_true: data.y_true, models };
    renderPredictionChart(chartData, checked);
    renderResidualChart(chartData, checked);

    const tbody = document.getElementById('metrics-body');
    tbody.innerHTML = checked.map(name => {
        const m = models[name]?.metrics;
        if (!m) return '';
        return `<tr>
            <td><span style="color:${COLORS[name]}">●</span> ${NAMES[name]}</td>
            <td>${(m.RMSE ?? 0).toFixed(4)}</td>
            <td>${(m.MAE ?? 0).toFixed(4)}</td>
            <td>${m.MAPE === null || m.MAPE === undefined ? '—' : Number(m.MAPE).toFixed(2)}</td>
            <td><strong>${(m.R2 ?? 0).toFixed(4)}</strong></td>
        </tr>`;
    }).join('');
}

function chartBase(dates) {
    return {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross', label: { backgroundColor: '#666' } },
            confine: true,
            textStyle: { fontSize: 12 },
        },
        legend: {
            top: 4,
            type: 'scroll',
            itemWidth: 14,
            itemHeight: 8,
            textStyle: { fontSize: 12 },
        },
        grid: { left: 56, right: 24, top: 44, bottom: 56 },
        toolbox: {
            right: 8,
            top: 0,
            itemSize: 14,
            feature: {
                dataZoom: { yAxisIndex: 'none', title: { zoom: '框选缩放', back: '还原缩放' } },
                restore: { title: '重置' },
                saveAsImage: { title: '保存图片', pixelRatio: 2 },
            },
        },
        xAxis: {
            type: 'category',
            data: dates,
            boundaryGap: false,
            axisLabel: {
                formatter: v => (v ? String(v).substring(2, 7) : ''),
                hideOverlap: true,
            },
            axisTick: { alignWithLabel: true },
        },
        dataZoom: [
            {
                type: 'inside',
                xAxisIndex: 0,
                filterMode: 'none',
                zoomOnMouseWheel: true,
                moveOnMouseMove: true,
            },
            {
                type: 'slider',
                xAxisIndex: 0,
                height: 22,
                bottom: 8,
                filterMode: 'none',
                showDetail: true,
                showDataShadow: true,
                brushSelect: true,
                borderColor: '#ddd',
                fillerColor: 'rgba(127,97,255,0.12)',
                handleStyle: { color: '#7b61ff' },
                dataBackground: {
                    lineStyle: { color: '#c5c5d2', width: 0.6 },
                    areaStyle: { color: 'rgba(100,100,140,0.08)' },
                },
            },
        ],
    };
}

/** 缩放后可见点变少时加粗线条，避免糊成一片 */
function bindZoomThicken(chart, seriesNames) {
    chart.off('datazoom');
    chart.on('datazoom', () => {
        const opt = chart.getOption();
        const dz = (opt.dataZoom || []).find(z => z.start !== undefined) || {};
        const span = (dz.end ?? 100) - (dz.start ?? 0);
        // 可见比例越小 → 线越粗
        let w = 1.2;
        if (span < 5) w = 2.8;
        else if (span < 15) w = 2.2;
        else if (span < 40) w = 1.6;
        const next = opt.series.map(s => {
            if (!seriesNames.has(s.name)) return s;
            return {
                ...s,
                lineStyle: {
                    ...(s.lineStyle || {}),
                    width: s.name === '观测值' ? w + 0.8 : w,
                    opacity: span < 15 ? 1 : 0.85,
                },
            };
        });
        chart.setOption({ series: next });
    });
}

function renderPredictionChart(data, selectedModels) {
    if (!predChart) {
        predChart = echarts.init(document.getElementById('prediction-chart'),
            null, { renderer: 'canvas' });
        window.addEventListener('resize', () => predChart.resize());
    }

    const names = new Set(['观测值']);
    const series = [{
        name: '观测值',
        type: 'line',
        symbol: 'none',
        sampling: 'lttb',
        data: data.dates.map((d, i) => [d, data.y_true[i]]),
        lineStyle: { width: 1.6, color: '#5b8ff9', opacity: 0.55 },
        itemStyle: { color: '#5b8ff9' },
        z: 2,
        emphasis: { lineStyle: { width: 2.4, opacity: 1 } },
    }];

    // 主模型画在最上层、更醒目；对照模型更淡
    const order = [...selectedModels].sort((a, b) => {
        if (a === 'e_cosine') return -1;
        if (b === 'e_cosine') return 1;
        return 0;
    });

    order.forEach(name => {
        const m = data.models[name];
        if (!m) return;
        names.add(m.display);
        const isMain = name === 'e_cosine';
        series.push({
            name: m.display,
            type: 'line',
            symbol: 'none',
            sampling: 'lttb',
            data: data.dates.map((d, i) => [d, m.y_pred[i]]),
            lineStyle: {
                width: isMain ? 2.2 : 1.2,
                color: m.color,
                opacity: isMain ? 1 : 0.75,
                cap: 'round',
            },
            itemStyle: { color: m.color },
            z: isMain ? 5 : 3,
            emphasis: { lineStyle: { width: isMain ? 3 : 2, opacity: 1 } },
        });
    });

    const option = {
        ...chartBase(data.dates),
        yAxis: {
            type: 'value',
            name: '水氡 (Bq/L)',
            scale: true,
            axisLabel: { formatter: v => v.toFixed(1) },
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.5 } },
        },
        series,
    };
    predChart.setOption(option, true);
    bindZoomThicken(predChart, names);
}

function renderResidualChart(data, selectedModels) {
    if (!residualChart) {
        residualChart = echarts.init(document.getElementById('residual-chart'),
            null, { renderer: 'canvas' });
        window.addEventListener('resize', () => residualChart.resize());
    }

    const names = new Set();
    const series = [];
    const order = [...selectedModels].sort((a, b) => {
        if (a === 'e_cosine') return -1;
        if (b === 'e_cosine') return 1;
        return 0;
    });

    order.forEach(name => {
        const m = data.models[name];
        if (!m) return;
        names.add(m.display);
        const isMain = name === 'e_cosine';
        series.push({
            name: m.display,
            type: 'line',
            symbol: 'none',
            sampling: 'lttb',
            data: data.dates.map((d, i) => {
                const y = data.y_true[i], p = m.y_pred[i];
                return (y == null || p == null) ? null : +(y - p).toFixed(4);
            }),
            lineStyle: {
                width: isMain ? 1.5 : 1,
                color: m.color,
                opacity: isMain ? 1 : 0.55,
            },
            itemStyle: { color: m.color },
            z: isMain ? 5 : 3,
        });
    });

    series.push({
        name: '零线',
        type: 'line',
        symbol: 'none',
        data: data.dates.map(d => [d, 0]),
        lineStyle: { width: 1, color: '#bbb', type: 'dashed' },
        itemStyle: { color: '#bbb' },
        z: 1,
    });

    residualChart.setOption({
        ...chartBase(data.dates),
        yAxis: {
            type: 'value',
            name: '残差 (Bq/L)',
            scale: true,
            splitLine: { lineStyle: { type: 'dashed', opacity: 0.5 } },
        },
        series,
    }, true);
    bindZoomThicken(residualChart, names);
}

init();
