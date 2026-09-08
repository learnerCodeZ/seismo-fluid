let predChart = null;
let residualChart = null;
let currentWindow = 60;
let currentVar = 'radon';
const MODEL_COLORS = {
    naive: '#999999',
    rf: '#b0b0b0',      // 恢复为原来的灰色
    lstm: '#91cc75',
    lstm_rf: '#fde88a',  // 浅黄
    cnn_lstm: '#d46060', // 深红
};

const MODEL_NAMES = {
    naive: 'Naive',
    rf: '随机森林',
    lstm: 'LSTM',
    lstm_rf: 'LSTM-RF',
    cnn_lstm: 'CNN-LSTM',
};

const VAR_NAMES = {
    radon: '水氡',
    level: '水位',
    radon_level: '水氡+水位',
    radon_level_weather: '水氡+水位+气象',
};

async function init() {
    // 变量选择
    const varSelect = document.getElementById('varSelect');
    if (varSelect) {
        varSelect.addEventListener('change', e => {
            currentVar = e.target.value;
            loadPredictions();
        });
    }

    // 窗口选择
    const windowSelect = document.getElementById('windowSelect');
    if (windowSelect) {
        windowSelect.addEventListener('change', e => {
            currentWindow = parseInt(e.target.value);
            loadPredictions();
        });
    }

    document.querySelectorAll('#model-checkboxes input').forEach(cb => {
        cb.addEventListener('change', loadPredictions);
    });
    await loadPredictions();
    await loadMetrics();
}

async function loadPredictions() {
    const checked = Array.from(document.querySelectorAll('#model-checkboxes input:checked'))
        .map(cb => cb.value);
    if (checked.length === 0) return;

    // 使用新的多变量 API
    const apiUrl = `/api/predictions/${currentVar}/${currentWindow}`;

    const res = await fetch(apiUrl);
    const data = await res.json();

    if (data.error) {
        console.warn(data.error);
        return;
    }

    // 适配新API格式
    const chartData = {
        dates: data.dates,
        y_true: data.y_true,
        models: {}
    };
    checked.forEach(model => {
        chartData.models[model] = {
            display: MODEL_NAMES[model] || model,
            color: MODEL_COLORS[model] || '#999',
            y_pred: data.y_pred
        };
    });

    renderPredictionChart(chartData, checked);
    renderResidualChart(chartData, checked);
}

function renderPredictionChart(data, selectedModels) {
    if (!predChart) {
        predChart = echarts.init(document.getElementById('prediction-chart'));
        window.addEventListener('resize', () => predChart.resize());
    }

    const series = [
        {
            name: '观测值',
            type: 'line',
            data: data.dates.map((d, i) => [d, data.y_true[i]]),
            symbol: 'none',
            lineStyle: { width: 2, color: '#8ab4e8' },
            itemStyle: { color: '#8ab4e8' },
            z: 10,
        }
    ];

    selectedModels.forEach(model => {
        const m = data.models[model];
        if (!m) return;
        series.push({
            name: m.display,
            type: 'line',
            data: data.dates.map((d, i) => [d, m.y_pred[i]]),
            symbol: 'none',
            lineStyle: { width: 2, color: m.color, type: 'solid' },
            itemStyle: { color: m.color },
        });
    });

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let tip = params[0].axisValue + '<br/>';
                params.forEach(p => {
                    tip += `${p.marker} ${p.seriesName}: ${p.value[1] !== null ? p.value[1].toFixed(3) : '-'}<br/>`;
                });
                tip += `<span style="color:#999;font-size:11px">* 变量: ${VAR_NAMES[currentVar] || currentVar} | 窗口: ${currentWindow}天</span>`;
                return tip;
            }
        },
        legend: {
            data: ['观测值', ...selectedModels.map(m => data.models[m]?.display || m)],
            top: 10,
        },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: data.dates,
            axisLabel: {
                formatter: function(val) {
                    return val ? val.substring(0, 10) : '';
                }
            },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '标准化值' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series,
    };

    predChart.setOption(option, true);
}

function renderResidualChart(data, selectedModels) {
    if (!residualChart) {
        residualChart = echarts.init(document.getElementById('residual-chart'));
        window.addEventListener('resize', () => residualChart.resize());
    }

    const series = [];
    selectedModels.forEach(model => {
        const m = data.models[model];
        if (!m) return;
        const residual = data.y_true.map((y, i) => {
            if (y === null || m.y_pred[i] === null) return null;
            return y - m.y_pred[i];
        });
        series.push({
            name: m.display,
            type: 'line',
            data: data.dates.map((d, i) => [d, residual[i]]),
            symbol: 'none',
            lineStyle: { width: 1.5, color: m.color },
            itemStyle: { color: m.color },
            connectNulls: false,
        });
    });

    series.push({
        name: '零线',
        type: 'line',
        data: data.dates.map(d => [d, 0]),
        symbol: 'none',
        lineStyle: { width: 1, color: '#ccc', type: 'dashed' },
        itemStyle: { color: '#ccc' },
    });

    const option = {
        tooltip: { trigger: 'axis' },
        legend: {
            data: [...selectedModels.map(m => data.models[m]?.display || m), '零线'],
            top: 10,
        },
        grid: { left: 60, right: 30, top: 50, bottom: 40 },
        xAxis: {
            type: 'category',
            data: data.dates,
            axisLabel: {
                formatter: function(val) {
                    return val ? val.substring(0, 10) : '';
                }
            },
            boundaryGap: false,
        },
        yAxis: { type: 'value', name: '残差' },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5 }
        ],
        series,
    };

    residualChart.setOption(option, true);
}

async function loadMetrics() {
    const apiUrl = `/api/predictions/${currentVar}/${currentWindow}?model=cnn_lstm`;
    const res = await fetch(apiUrl);
    const data = await res.json();

    if (data.metrics) {
        const tbody = document.getElementById('metrics-body');
        tbody.innerHTML = `
            <tr>
                <td><span style="color:${MODEL_COLORS.cnn_lstm}">&#9679;</span> CNN-LSTM</td>
                <td>${data.metrics.RMSE.toFixed(4)}</td>
                <td>${data.metrics.MAE.toFixed(4)}</td>
                <td>${data.metrics.MAPE.toFixed(2)}%</td>
                <td>${data.metrics.R2.toFixed(4)}</td>
            </tr>
        `;
    }
}

init();
