/**
 * 金箔锻制工艺仿真系统 - 前端核心逻辑
 * 包含: Three.js 3D可视化, Canvas厚度云图, WebSocket实时通信
 */

const API_BASE = window.location.origin;
const WS_URL = window.location.origin.replace('http', 'ws') + '/ws';

let scene, camera, renderer, controls, foilMesh, hammerMesh;
let gridSize = 48;
let foilSize = 150;
let currentThicknessData = null;
let autoMode = false;
let autoIntervalId = null;
let strikeHistoryData = [];
let alertsData = [];
let selectedColormap = 'turbo';

const COLORMAPS = {
    viridis: (t) => {
        const c = [
            [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
            [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
            [180, 222, 44], [253, 231, 37]
        ];
        return sampleColor(c, t);
    },
    turbo: (t) => {
        const c = [
            [48, 18, 59], [68, 28, 142], [62, 59, 219], [31, 106, 247],
            [19, 161, 218], [27, 206, 164], [85, 242, 100], [171, 252, 53],
            [237, 239, 48], [250, 176, 49], [240, 113, 48], [218, 55, 53],
            [173, 19, 62]
        ];
        return sampleColor(c, t);
    },
    jet: (t) => {
        if (t < 0.125) return [0, 0, 128 + t * 1024];
        if (t < 0.375) return [0, (t - 0.125) * 1024, 255];
        if (t < 0.625) return [(t - 0.375) * 1024, 255, 255 - (t - 0.375) * 1024];
        if (t < 0.875) return [255, 255 - (t - 0.625) * 1024, 0];
        return [255 - (t - 0.875) * 1024, 0, 0];
    },
    thermal: (t) => {
        const c = [
            [0, 0, 0], [40, 0, 40], [120, 0, 120], [200, 20, 80],
            [255, 80, 0], [255, 160, 0], [255, 230, 80], [255, 255, 255]
        ];
        return sampleColor(c, t);
    }
};

function sampleColor(colors, t) {
    t = Math.max(0, Math.min(1, t));
    const idx = t * (colors.length - 1);
    const i = Math.floor(idx);
    const f = idx - i;
    if (i >= colors.length - 1) return colors[colors.length - 1];
    return [
        Math.round(colors[i][0] + (colors[i+1][0] - colors[i][0]) * f),
        Math.round(colors[i][1] + (colors[i+1][1] - colors[i][1]) * f),
        Math.round(colors[i][2] + (colors[i+1][2] - colors[i][2]) * f),
    ];
}

function getColormapColor(value, min, max) {
    const t = (value - min) / (max - min + 1e-8);
    const cm = COLORMAPS[selectedColormap] || COLORMAPS.turbo;
    const rgb = cm(t);
    return rgb;
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

function showToast(title, message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
    `;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function initThreeJS() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = null;

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    camera.position.set(0, 180, 200);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(100, 200, 100);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0xffd700, 0.8, 500);
    pointLight.position.set(-50, 80, -50);
    scene.add(pointLight);

    createFoilMesh();
    createHammerMesh();

    const gridHelper = new THREE.GridHelper(foilSize * 1.5, 20, 0x333333, 0x222222);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    window.addEventListener('resize', onWindowResize);
    animate();
}

function createFoilMesh() {
    const geometry = new THREE.PlaneGeometry(foilSize, foilSize, gridSize - 1, gridSize - 1);
    geometry.rotateX(-Math.PI / 2);

    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 100,
        specular: 0x222222,
    });

    foilMesh = new THREE.Mesh(geometry, material);
    foilMesh.receiveShadow = true;
    foilMesh.castShadow = true;
    scene.add(foilMesh);

    updateFoilColorsUniform();
}

function createHammerMesh() {
    const handleGeo = new THREE.CylinderGeometry(2, 2, 60, 16);
    const handleMat = new THREE.MeshPhongMaterial({ color: 0x8B4513, shininess: 20 });
    const handle = new THREE.Mesh(handleGeo, handleMat);
    
    const headGeo = new THREE.CylinderGeometry(10, 10, 20, 16);
    const headMat = new THREE.MeshPhongMaterial({ color: 0x444444, shininess: 80 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = -30;

    hammerMesh = new THREE.Group();
    hammerMesh.add(handle);
    hammerMesh.add(head);
    hammerMesh.position.set(0, 80, 0);
    hammerMesh.rotation.z = Math.PI / 6;
    scene.add(hammerMesh);
}

function updateFoilColorsUniform() {
    if (!foilMesh) return;
    const geometry = foilMesh.geometry;
    const colors = [];
    const defaultColor = new THREE.Color(0xd4af37);
    
    for (let i = 0; i < geometry.attributes.position.count; i++) {
        colors.push(defaultColor.r, defaultColor.g, defaultColor.b);
    }
    
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.attributes.color.needsUpdate = true;
}

function updateFoilVisualization(thicknessData) {
    if (!foilMesh || !thicknessData) return;
    
    currentThicknessData = thicknessData;
    const { thickness_um, min_um, max_um, grid_size } = thicknessData;
    gridSize = grid_size || 48;

    const geometry = foilMesh.geometry;
    const positions = geometry.attributes.position;
    const colors = geometry.attributes.color;

    for (let i = 0; i < positions.count; i++) {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        if (row < thickness_um.length && col < thickness_um[row].length) {
            const t = thickness_um[row][col];
            
            const heightScale = Math.max(0.1, Math.min(t / (max_um + 1e-8), 1));
            const visualHeight = -2 + heightScale * 4;
            positions.setY(i, visualHeight);

            const rgb = getColormapColor(t, min_um, max_um);
            const useColor = document.getElementById('toggle-color').checked;
            if (useColor) {
                colors.setXYZ(i, rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
            } else {
                colors.setXYZ(i, 212/255, 175/255, 55/255);
            }
        }
    }

    positions.needsUpdate = true;
    colors.needsUpdate = true;
    geometry.computeVertexNormals();

    foilMesh.material.wireframe = document.getElementById('toggle-wireframe').checked;

    document.getElementById('legend-min').textContent = min_um.toFixed(2);
    document.getElementById('legend-max').textContent = max_um.toFixed(2);

    drawHeatmapCanvas(thicknessData);
}

function animateHammerStrike(position, force) {
    if (!hammerMesh) return;
    
    const startPos = { y: 80, rx: Math.PI / 6 };
    const endPos = { y: 10, rx: 0 };
    const duration = 300;
    const startTime = Date.now();

    hammerMesh.position.x = position[0];
    hammerMesh.position.z = position[1];

    function tick() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        let eased;
        if (progress < 0.5) {
            eased = progress * 2;
            eased = eased * eased;
        } else {
            eased = 1 - (progress - 0.5) * 2;
            eased = 1 - eased * eased;
        }

        hammerMesh.position.y = startPos.y + (endPos.y - startPos.y) * eased;
        hammerMesh.rotation.z = startPos.rx + (endPos.rx - startPos.rx) * eased;
        hammerMesh.rotation.x = Math.sin(elapsed * 0.05) * 0.05;

        if (progress < 1) {
            requestAnimationFrame(tick);
        } else {
            setTimeout(() => {
                hammerMesh.position.set(0, 80, 0);
                hammerMesh.rotation.z = Math.PI / 6;
            }, 100);
        }
    }
    tick();
}

function animate() {
    requestAnimationFrame(animate);
    
    if (controls && document.getElementById('toggle-auto-rotate').checked) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
    } else if (controls) {
        controls.autoRotate = false;
    }
    
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    if (!container || !camera || !renderer) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function drawHeatmapCanvas(thicknessData) {
    const canvas = document.getElementById('heatmap-canvas');
    if (!canvas || !thicknessData) return;

    const { thickness_um, min_um, max_um } = thicknessData;
    const grid = thickness_um.length;
    const gs = grid || 48;

    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width - 24, 350);
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext('2d');
    const cellSize = size / gs;

    ctx.clearRect(0, 0, size, size);

    for (let i = 0; i < gs; i++) {
        for (let j = 0; j < gs; j++) {
            const value = thickness_um[i][j];
            const rgb = getColormapColor(value, min_um, max_um);
            ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize + 1, cellSize + 1);
        }
    }

    ctx.strokeStyle = 'rgba(212, 175, 55, 0.3)';
    ctx.lineWidth = 0.5;
    const gridLines = 8;
    for (let i = 0; i <= gridLines; i++) {
        const p = (i / gridLines) * size;
        ctx.beginPath();
        ctx.moveTo(p, 0);
        ctx.lineTo(p, size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, p);
        ctx.lineTo(size, p);
        ctx.stroke();
    }

    canvas._thicknessData = thicknessData;
}

function setupCanvasTooltip() {
    const canvas = document.getElementById('heatmap-canvas');
    const tooltip = document.getElementById('canvas-tooltip');
    if (!canvas) return;

    canvas.addEventListener('mousemove', (e) => {
        if (!canvas._thicknessData) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const { thickness_um, min_um, max_um, grid_size } = canvas._thicknessData;
        const gs = grid_size || 48;
        const row = Math.floor((y / canvas.height) * gs);
        const col = Math.floor((x / canvas.width) * gs);
        
        if (row >= 0 && row < gs && col >= 0 && col < gs) {
            const value = thickness_um[row][col];
            const x_mm = ((col / gs) - 0.5) * foilSize;
            const y_mm = ((row / gs) - 0.5) * foilSize;
            const deviation = (value - (min_um + max_um) / 2) / ((max_um - min_um) / 2 + 1e-8) * 100;

            tooltip.style.display = 'block';
            tooltip.style.left = (e.pageX + 12) + 'px';
            tooltip.style.top = (e.pageY + 12) + 'px';
            tooltip.innerHTML = `
                <div>位置: <span class="value">(${x_mm.toFixed(1)}, ${y_mm.toFixed(1)}) mm</span></div>
                <div>厚度: <span class="value">${value.toFixed(4)} μm</span></div>
                <div>归一化: <span class="value">${deviation > 0 ? '+' : ''}${deviation.toFixed(1)}%</span></div>
            `;
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

function updateMetricsDisplay(state) {
    const metrics = state.thickness_distribution?.metrics || {};
    
    document.getElementById('metric-strikes').textContent = state.total_strikes || 0;
    document.getElementById('metric-thickness').textContent = 
        (metrics.mean_thickness_um || 500).toFixed(2) + ' μm';
    document.getElementById('metric-cv').textContent = 
        (metrics.coefficient_of_variation || 0).toFixed(4);
    document.getElementById('metric-elongation').textContent = 
        (state.total_elongation || 1).toFixed(2) + 'x';
    document.getElementById('metric-temp').textContent = 
        (state.temperature_c || 25).toFixed(1) + '°C';
    document.getElementById('metric-strain').textContent = 
        (state.plastic_strain || 0).toFixed(3);

    document.getElementById('thickness-min').textContent = 
        (metrics.min_thickness_um || 500).toFixed(2) + ' μm';
    document.getElementById('thickness-max').textContent = 
        (metrics.max_thickness_um || 500).toFixed(2) + ' μm';

    const u10 = (metrics.uniformity_within_10pct || 1);
    document.getElementById('uniformity-fill').style.width = (u10 * 100) + '%';
    document.getElementById('uniformity-value').textContent = (u10 * 100).toFixed(1) + '%';

    document.getElementById('trend-reward').textContent = 
        (state.rl_stats?.avg_reward || 0).toFixed(2);
    document.getElementById('trend-epsilon').textContent = 
        (state.rl_stats?.epsilon || 1).toFixed(2);
    document.getElementById('trend-u5').textContent = 
        ((metrics.uniformity_within_5pct || 1) * 100).toFixed(0) + '%';
    document.getElementById('trend-range').textContent = 
        ((metrics.range_ratio || 0) * 100).toFixed(1) + '%';
}

function updateRiskDisplay(risk) {
    const indicator = document.getElementById('risk-indicator');
    const title = indicator.querySelector('.risk-title');
    const detail = indicator.querySelector('.risk-detail');

    const classes = ['risk-none', 'risk-low', 'risk-medium', 'risk-high'];
    classes.forEach(c => indicator.classList.remove(c));
    indicator.classList.add('risk-' + (risk.risk_level || 'none'));

    const titles = {
        none: '✅ 无风险',
        low: '⚠️ 低风险',
        medium: '🔶 中风险',
        high: '🛑 高风险'
    };
    title.textContent = titles[risk.risk_level] || titles.none;

    const details = {
        none: '厚度分布正常，所有点均高于阈值',
        low: '局部区域接近阈值，建议加强监控',
        medium: '局部厚度低于阈值，注意控制锤击力度',
        high: '严重低于阈值，立即停止并退火处理'
    };
    detail.textContent = risk.min_thickness_um !== undefined 
        ? `最薄点: ${risk.min_thickness_um.toFixed(4)}μm | 风险点: ${risk.risk_count || 0}个`
        : details[risk.risk_level] || '';
}

function addAlertItem(alert) {
    if (alertsData.length === 0) {
        document.querySelector('#alerts-log .log-empty')?.remove();
    }
    alertsData.unshift(alert);
    if (alertsData.length > 50) alertsData.pop();

    const log = document.getElementById('alerts-log');
    const item = document.createElement('div');
    item.className = `alert-item alert-${alert.level || 'low'}`;
    
    const time = new Date(alert.timestamp).toLocaleTimeString('zh-CN');
    item.innerHTML = `
        <div><strong>[${time}]</strong> ${alert.message || ''}</div>
        <div style="margin-top:3px;opacity:0.8;">最薄: ${(alert.risk?.min_thickness_um || 0).toFixed(4)}μm</div>
    `;
    log.insertBefore(item, log.firstChild);

    const types = { high: 'error', medium: 'warning', low: 'warning' };
    showToast(`⚠️ 破裂${alert.level?.toUpperCase()}预警`, 
        alert.message || '厚度异常', types[alert.level] || 'warning');
}

function addStrikeHistoryItem(result) {
    const strike = result.strike || result;
    if (strikeHistoryData.length === 0) {
        document.querySelector('#strike-history .log-empty')?.remove();
    }
    strikeHistoryData.unshift(strike);
    if (strikeHistoryData.length > 100) strikeHistoryData.pop();

    const log = document.getElementById('strike-history');
    const item = document.createElement('div');
    item.className = 'strike-item';
    
    const num = strike.strike_num || strikeHistoryData.length;
    const force = strike.hammer_force_N || result.action?.force_N || 0;
    const thick = strike.avg_thickness_um || strike.metrics?.mean_thickness_um || 0;
    const pos = strike.hammer_position || result.action?.position_mm || [0, 0];
    
    item.innerHTML = `
        <span class="strike-num">#${num}</span>
        <span class="strike-info">${force.toFixed(0)}N<br>(${pos[0].toFixed(0)},${pos[1].toFixed(0)})</span>
        <span class="strike-thickness">${thick.toFixed(2)}μm</span>
    `;
    log.insertBefore(item, log.firstChild);
}

let ws = null;
let wsReconnectTimer = null;

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
                
                if (data.alert) addAlertItem(data.alert);
                if (data.strike || data.action) addStrikeHistoryItem(data);
                
                const state = data.thickness_distribution ? data : 
                              (await fetchState());
                              
                if (state) {
                    updateMetricsDisplay(state);
                    updateRiskDisplay(state.fracture_risk || data.fracture_risk || {});
                    
                    if (state.thickness_distribution) {
                        const viz = await fetchThicknessViz();
                        if (viz) updateFoilVisualization(viz);
                    }
                }
                
                if (data.action) {
                    animateHammerStrike(data.action.position_mm, data.action.force_N);
                }
            } 
            else if (msg.channel === 'alerts') {
                addAlertItem(msg.data);
            }
            else if (msg.channel === 'thickness_viz') {
                updateFoilVisualization(msg.data);
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

async function fetchJSON(url, options) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error('[API] 请求失败:', url, e);
        return null;
    }
}

async function fetchState() {
    return await fetchJSON(API_BASE + '/api/state');
}

async function fetchThicknessViz() {
    return await fetchJSON(API_BASE + '/api/visualization/thickness');
}

async function performStrike() {
    const mode = document.getElementById('strike-mode').value;
    let result;
    
    if (mode === 'manual') {
        const payload = {
            force_N: parseFloat(document.getElementById('force-slider').value),
            position_x_mm: parseFloat(document.getElementById('posx-slider').value),
            position_y_mm: parseFloat(document.getElementById('posy-slider').value),
            radius_mm: 15,
        };
        result = await fetchJSON(API_BASE + '/api/strike', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        animateHammerStrike([payload.position_x_mm, payload.position_y_mm], payload.force_N);
    } else {
        const rlMode = mode === 'rl' ? 'rl' : 'heuristic';
        result = await fetchJSON(API_BASE + `/api/strike/auto?mode=${rlMode}`, {
            method: 'POST'
        });
        if (result?.action) {
            animateHammerStrike(result.action.position_mm, result.action.force_N);
        }
    }
    
    if (result) {
        if (result.alert) addAlertItem(result.alert);
        addStrikeHistoryItem(result);
        
        const state = await fetchState();
        if (state) {
            updateMetricsDisplay(state);
            updateRiskDisplay(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) updateFoilVisualization(viz);
    }
}

async function startAutoSimulation() {
    if (autoMode) return;
    autoMode = true;
    
    document.getElementById('btn-auto').style.display = 'none';
    document.getElementById('btn-stop').style.display = 'flex';
    
    const mode = document.getElementById('strike-mode').value;
    const rlMode = mode === 'rl' ? 'rl' : 'heuristic';
    const interval = parseFloat(document.getElementById('interval-slider').value);
    
    const result = await fetchJSON(
        API_BASE + `/api/simulation/auto/start?interval_sec=${interval}&mode=${rlMode}`,
        { method: 'POST' }
    );
    
    if (result) {
        showToast('✅ 自动锻制已启动', 
            `模式: ${rlMode} | 间隔: ${interval}s`, 'success');
    }
}

async function stopAutoSimulation() {
    autoMode = false;
    document.getElementById('btn-auto').style.display = 'flex';
    document.getElementById('btn-stop').style.display = 'none';
    
    await fetchJSON(API_BASE + '/api/simulation/auto/stop', { method: 'POST' });
    showToast('⏹ 已停止', '自动锻制已停止', 'info');
}

async function performAnnealing() {
    const temp = parseFloat(document.getElementById('anneal-temp').value);
    const result = await fetchJSON(API_BASE + '/api/anneal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature_c: temp, duration_min: 10 }),
    });
    
    if (result) {
        showToast('🔥 退火已执行', 
            result.annealing?.message || `退火温度:${temp}°C`, 'warning');
        const state = await fetchState();
        if (state) {
            updateMetricsDisplay(state);
            updateRiskDisplay(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) updateFoilVisualization(viz);
    }
}

async function resetSimulation() {
    if (!confirm('确定要重置仿真吗？所有进度将丢失。')) return;
    
    const result = await fetchJSON(API_BASE + '/api/reset', { method: 'POST' });
    if (result) {
        showToast('↺ 已重置', '仿真已恢复初始状态', 'info');
        strikeHistoryData = [];
        alertsData = [];
        document.getElementById('strike-history').innerHTML = 
            '<div class="log-empty">等待锤击...</div>';
        document.getElementById('alerts-log').innerHTML = 
            '<div class="log-empty">暂无告警</div>';
            
        const state = await fetchState();
        if (state) {
            updateMetricsDisplay(state);
            updateRiskDisplay(state.fracture_risk || {});
        }
        const viz = await fetchThicknessViz();
        if (viz) updateFoilVisualization(viz);
    }
}

function exportJSON() {
    const data = {
        export_time: new Date().toISOString(),
        strike_history: strikeHistoryData,
        alerts: alertsData,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gold-foil-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('💾 导出成功', '历史数据已导出为JSON', 'success');
}

function exportCSV() {
    if (strikeHistoryData.length === 0) {
        showToast('⚠️ 无数据', '没有可导出的历史数据', 'warning');
        return;
    }
    
    const headers = ['strike_num', 'hammer_force_N', 'pos_x', 'pos_y', 
                     'avg_thickness_um', 'min_um', 'max_um', 'std_um', 
                     'elongation_rate', 'cv'];
    const rows = [headers.join(',')];
    
    for (const s of strikeHistoryData) {
        const pos = s.hammer_position || [0, 0];
        const cv = (s.thickness_std_um || 0) / (s.avg_thickness_um || 1);
        rows.push([
            s.strike_num || 0,
            s.hammer_force_N || 0,
            pos[0] || 0,
            pos[1] || 0,
            s.avg_thickness_um || 0,
            s.min_thickness_um || 0,
            s.max_thickness_um || 0,
            s.thickness_std_um || 0,
            s.elongation_rate || 0,
            cv.toFixed(6),
        ].join(','));
    }
    
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gold-foil-metrics-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('💾 导出成功', '指标数据已导出为CSV', 'success');
}

async function checkHealth() {
    try {
        const res = await fetch(API_BASE + '/api/health');
        const data = await res.json();
        const influxEl = document.getElementById('influx-status');
        influxEl.textContent = 'InfluxDB: ' + (data.influxdb === 'connected' ? '已连接' : '未连接');
        influxEl.className = 'status-badge ' + 
            (data.influxdb === 'connected' ? 'status-connected' : 'status-disconnected');
    } catch (e) {}
}

function setupEventListeners() {
    document.getElementById('btn-strike').addEventListener('click', performStrike);
    document.getElementById('btn-auto').addEventListener('click', startAutoSimulation);
    document.getElementById('btn-stop').addEventListener('click', stopAutoSimulation);
    document.getElementById('btn-anneal').addEventListener('click', performAnnealing);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);
    document.getElementById('btn-export').addEventListener('click', exportJSON);
    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);

    const updateSliderLabel = (sliderId, labelId, suffix = '') => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider && label) {
            const update = () => { label.textContent = slider.value + suffix; };
            slider.addEventListener('input', update);
            update();
        }
    };
    
    updateSliderLabel('force-slider', 'force-label');
    updateSliderLabel('posx-slider', 'posx-label');
    updateSliderLabel('posy-slider', 'posy-label');
    updateSliderLabel('interval-slider', 'interval-label');
    updateSliderLabel('anneal-temp', 'anneal-temp-label');

    document.getElementById('strike-mode').addEventListener('change', (e) => {
        const manual = document.getElementById('manual-controls');
        manual.style.display = e.target.value === 'manual' ? 'block' : 'none';
    });

    document.getElementById('colormap-select').addEventListener('change', (e) => {
        selectedColormap = e.target.value;
        if (currentThicknessData) {
            updateFoilVisualization(currentThicknessData);
            drawHeatmapCanvas(currentThicknessData);
        }
    });

    ['toggle-wireframe', 'toggle-color', 'toggle-auto-rotate'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            if (currentThicknessData) updateFoilVisualization(currentThicknessData);
        });
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') { e.preventDefault(); performStrike(); }
        if (e.code === 'KeyR') resetSimulation();
        if (e.code === 'KeyA') autoMode ? stopAutoSimulation() : startAutoSimulation();
    });
}

async function init() {
    setupEventListeners();
    initThreeJS();
    setupCanvasTooltip();
    
    checkHealth();
    setInterval(checkHealth, 30000);
    
    const initialViz = await fetchThicknessViz();
    if (initialViz) updateFoilVisualization(initialViz);
    
    const initialState = await fetchState();
    if (initialState) {
        updateMetricsDisplay(initialState);
        updateRiskDisplay(initialState.fracture_risk || {});
    }
    
    connectWebSocket();
    
    console.log('%c🏺 金箔锻制工艺仿真系统已启动', 
        'color:#d4af37;font-size:16px;font-weight:bold');
    console.log('快捷键: [空格]锤击 | [A]自动/停止 | [R]重置');
}

document.addEventListener('DOMContentLoaded', init);
