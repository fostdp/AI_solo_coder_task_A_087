/**
 * 金箔锻制工艺仿真系统 - 前端主入口 v3 (模块化架构)
 *
 * 模块:
 *   - GoldFoil3D:   三维渲染 (GPU Shader)
 *   - ThicknessPanel: UI控制面板 + 厚度云图
 *
 * 本文件:
 *   - WebSocket 连接管理
 *   - API 调用封装
 *   - 协调两个模块的状态同步
 *   - 事件回调桥接
 */

const API_BASE = window.location.origin;
const WS_URL = window.location.origin.replace('http', 'ws') + '/ws';

let ws = null;
let wsReconnectTimer = null;
let autoMode = false;
let autoIntervalId = null;


function connectWebSocket() {
    const statusEl = document.getElementById('connection-status');
    statusEl.textContent = '连接中...';
    statusEl.className = 'status-badge status-connecting';

    try {
        ws = new WebSocket(WS_URL);
    } catch (e) {
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        statusEl.textContent = '● WebSocket已连接';
        statusEl.className = 'status-badge status-connected';
        console.log('[WS] 连接已建立');

        ws.send(JSON.stringify({ type: 'get_state' }));

        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
    };

    ws.onmessage = async (event) => {
        try {
            const msg = JSON.parse(event.data);

            if (msg.channel === 'state_update') {
                const data = msg.data;

                if (data.alert) {
                    ThicknessPanel.addAlert(data.alert);
                }
                if (data.strike || data.action) {
                    ThicknessPanel.addStrikeHistory(data);
                }

                const hasThickness = data.thickness_distribution || data.metrics;
                const state = hasThickness ? (await fetchState()) : null;

                if (state) {
                    ThicknessPanel.updateMetrics(state);
                    ThicknessPanel.updateRisk(state.fracture_risk || data.fracture_risk || {});

                    if (state.thickness_distribution || data.thickness_distribution) {
                        const viz = await fetchThicknessViz();
                        if (viz) {
                            GoldFoil3D.updateThickness(viz);
                            ThicknessPanel.updateHeatmap(viz);
                        }
                    }
                }

                if (data.action) {
                    GoldFoil3D.animateStrike(
                        data.action.position_mm || data.action.position,
                        data.action.force_N || data.action.force
                    );
                }
            }
            else if (msg.channel === 'alerts') {
                ThicknessPanel.addAlert(msg.data);
            }
            else if (msg.channel === 'thickness_viz') {
                GoldFoil3D.updateThickness(msg.data);
                ThicknessPanel.updateHeatmap(msg.data);
            }
            else if (msg.type === 'connected') {
                console.log('[WS] 确认连接', msg);
            }
        } catch (e) {
            console.error('[WS] 消息解析错误:', e, event.data);
        }
    };

    ws.onclose = () => {
        statusEl.textContent = '⚠ WebSocket已断开';
        statusEl.className = 'status-badge status-disconnected';
        scheduleReconnect();
    };

    ws.onerror = (err) => {
        console.error('[WS] 错误:', err);
        statusEl.textContent = '连接错误';
        statusEl.className = 'status-badge status-disconnected';
    };
}

function scheduleReconnect() {
    if (wsReconnectTimer) return;
    wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectWebSocket();
    }, 3000);
}


async function fetchState() {
    return await ThicknessPanel.fetchJSON(API_BASE + '/api/state');
}

async function fetchThicknessViz() {
    return await ThicknessPanel.fetchJSON(API_BASE + '/api/visualization/thickness');
}

async function performStrike() {
    const mode = ThicknessPanel.getStrikeMode();
    let result;

    if (mode === 'manual') {
        const params = ThicknessPanel.getManualParams();
        result = await ThicknessPanel.fetchJSON(API_BASE + '/api/strike', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        GoldFoil3D.animateStrike(
            [params.position_x_mm, params.position_y_mm],
            params.force_N
        );
    } else {
        const rlMode = mode === 'rl' ? 'pretrained' :
            (mode === 'rl_heuristic' ? 'heuristic' : mode);
        result = await ThicknessPanel.fetchJSON(
            API_BASE + `/api/strike/auto?mode=${rlMode}`,
            { method: 'POST' }
        );
        if (result?.action) {
            GoldFoil3D.animateStrike(
                result.action.position_mm || result.action.position,
                result.action.force_N || result.action.force
            );
        }
    }

    if (result) {
        if (result.alert) ThicknessPanel.addAlert(result.alert);
        ThicknessPanel.addStrikeHistory(result);

        const state = await fetchState();
        if (state) {
            ThicknessPanel.updateMetrics(state);
            ThicknessPanel.updateRisk(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) {
            GoldFoil3D.updateThickness(viz);
            ThicknessPanel.updateHeatmap(viz);
        }
    }
}

async function startAutoSimulation() {
    if (autoMode) return;
    autoMode = true;

    document.getElementById('btn-auto').style.display = 'none';
    document.getElementById('btn-stop').style.display = 'flex';

    const mode = ThicknessPanel.getStrikeMode();
    const rlMode = mode === 'rl' ? 'pretrained' :
        (mode === 'rl_heuristic' ? 'heuristic' : mode);
    const interval = ThicknessPanel.getAutoInterval();

    const result = await ThicknessPanel.fetchJSON(
        API_BASE + `/api/simulation/auto/start?interval_sec=${interval}&mode=${rlMode}`,
        { method: 'POST' }
    );

    if (result) {
        ThicknessPanel.showToast(
            '✅ 自动锻制已启动',
            `模式: ${rlMode} | 间隔: ${interval}s`,
            'success'
        );
    }
}

async function stopAutoSimulation() {
    autoMode = false;
    document.getElementById('btn-auto').style.display = 'flex';
    document.getElementById('btn-stop').style.display = 'none';

    await ThicknessPanel.fetchJSON(API_BASE + '/api/simulation/auto/stop',
        { method: 'POST' });
    ThicknessPanel.showToast('⏹ 已停止', '自动锻制已停止', 'info');
}

async function performAnnealing() {
    const temp = ThicknessPanel.getAnnealTemp();
    const result = await ThicknessPanel.fetchJSON(API_BASE + '/api/anneal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature_c: temp, duration_min: 10 }),
    });

    if (result) {
        ThicknessPanel.showToast(
            '🔥 退火已执行',
            result.annealing?.message || `退火温度:${temp}°C`,
            'warning'
        );
        const state = await fetchState();
        if (state) {
            ThicknessPanel.updateMetrics(state);
            ThicknessPanel.updateRisk(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) {
            GoldFoil3D.updateThickness(viz);
            ThicknessPanel.updateHeatmap(viz);
        }
    }
}

async function resetSimulation() {
    if (!confirm('确定要重置仿真吗？所有进度将丢失。')) return;

    const result = await ThicknessPanel.fetchJSON(API_BASE + '/api/reset',
        { method: 'POST' });
    if (result) {
        ThicknessPanel.showToast('↺ 已重置', '仿真已恢复初始状态', 'info');
        ThicknessPanel.clearHistory();

        const state = await fetchState();
        if (state) {
            ThicknessPanel.updateMetrics(state);
            ThicknessPanel.updateRisk(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) {
            GoldFoil3D.updateThickness(viz);
            ThicknessPanel.updateHeatmap(viz);
        }
    }
}

function exportJSONData() {
    ThicknessPanel.exportJSON();
}

function exportCSVData() {
    ThicknessPanel.exportCSV();
}

function onColormapChange(name) {
    GoldFoil3D.setColormap(name);
    if (ThicknessPanel.currentThicknessData) {
        ThicknessPanel.updateHeatmap(ThicknessPanel.currentThicknessData);
    }
}

function onDisplayChange(opts) {
    GoldFoil3D.setWireframe(opts.wireframe);
    GoldFoil3D.setColorEnabled(opts.color);
    GoldFoil3D.setAutoRotate(opts.autoRotate);
}

async function checkHealth() {
    try {
        const res = await fetch(API_BASE + '/api/health');
        const data = await res.json();
        const influxEl = document.getElementById('influx-status');
        if (influxEl) {
            influxEl.textContent = 'InfluxDB: ' + (data.influxdb === 'connected' ? '已连接' : '未连接');
            influxEl.className = 'status-badge ' +
                (data.influxdb === 'connected' ? 'status-connected' : 'status-disconnected');
        }
    } catch (e) { }
}


async function init() {
    GoldFoil3D.init('three-container', { foilSize: 150 });
    ThicknessPanel.init({ foilSize: 150 });

    ThicknessPanel.setEventCallbacks({
        onStrike: performStrike,
        onAutoStart: startAutoSimulation,
        onAutoStop: stopAutoSimulation,
        onAnneal: performAnnealing,
        onReset: resetSimulation,
        onExportJSON: exportJSONData,
        onExportCSV: exportCSVData,
        onColormapChange: onColormapChange,
        onDisplayChange: onDisplayChange,
    });

    checkHealth();
    setInterval(checkHealth, 30000);

    const initialViz = await fetchThicknessViz();
    if (initialViz) {
        GoldFoil3D.updateThickness(initialViz);
        ThicknessPanel.updateHeatmap(initialViz);
    }

    const initialState = await fetchState();
    if (initialState) {
        ThicknessPanel.updateMetrics(initialState);
        ThicknessPanel.updateRisk(initialState.fracture_risk || {});
    }

    connectWebSocket();

    console.log('%c🏺 金箔锻制工艺仿真系统已启动 (v3 模块化)',
        'color:#d4af37;font-size:16px;font-weight:bold');
    console.log('模块: GoldFoil3D + ThicknessPanel');
    console.log('渲染模式:', GoldFoil3D.isMobile ? '移动端 (低分辨率+无阴影)' : '桌面端 (高保真)');
    console.log('快捷键: [空格]锤击 | [A]自动/停止 | [R]重置');
}

document.addEventListener('DOMContentLoaded', init);
