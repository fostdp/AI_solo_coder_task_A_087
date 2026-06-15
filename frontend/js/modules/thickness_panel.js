/**
 * ThicknessPanel - 厚度云图面板与UI控制模块
 *
 * 职责:
 *   - 2D Canvas 厚度热力图绘制
 *   - 指标卡显示更新
 *   - 风险指示卡显示
 *   - 告警日志 / 锤击历史
 *   - UI 控件事件绑定与回调
 *   - API 请求封装
 *
 * 对外 API:
 *   ThicknessPanel.init(options)
 *   ThicknessPanel.updateMetrics(state)
 *   ThicknessPanel.updateRisk(risk)
 *   ThicknessPanel.updateHeatmap(thicknessData)
 *   ThicknessPanel.addAlert(alert)
 *   ThicknessPanel.addStrikeHistory(result)
 *   ThicknessPanel.setEventCallbacks(callbacks)
 *   ThicknessPanel.fetchJSON(url, options)
 */

const ThicknessPanel = (function () {
    "use strict";

    let selectedColormap = 'turbo';
    let currentThicknessData = null;
    let strikeHistoryData = [];
    let alertsData = [];
    let isMobileFlag = false;
    let foilSize = 150;

    const COLORMAP_LUT_SIZE = 512;

    const COLORMAPS = {
        viridis: (t) => {
            const c = [
                [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
                [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
                [180, 222, 44], [253, 231, 37]
            ];
            return _sampleColor(c, t);
        },
        turbo: (t) => {
            const c = [
                [48, 18, 59], [68, 28, 142], [62, 59, 219], [31, 106, 247],
                [19, 161, 218], [27, 206, 164], [85, 242, 100], [171, 252, 53],
                [237, 239, 48], [250, 176, 49], [240, 113, 48], [218, 55, 53],
                [173, 19, 62]
            ];
            return _sampleColor(c, t);
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
            return _sampleColor(c, t);
        }
    };

    let callbacks = {
        onStrike: null,
        onAutoStart: null,
        onAutoStop: null,
        onAnneal: null,
        onReset: null,
        onExportJSON: null,
        onExportCSV: null,
        onColormapChange: null,
    };

    function _sampleColor(colors, t) {
        t = Math.max(0, Math.min(1, t));
        const idx = t * (colors.length - 1);
        const i = Math.floor(idx);
        const f = idx - i;
        if (i >= colors.length - 1) return colors[colors.length - 1];
        return [
            Math.round(colors[i][0] + (colors[i + 1][0] - colors[i][0]) * f),
            Math.round(colors[i][1] + (colors[i + 1][1] - colors[i][1]) * f),
            Math.round(colors[i][2] + (colors[i + 1][2] - colors[i][2]) * f),
        ];
    }

    function _getColormapColor(value, min, max) {
        const t = (value - min) / (max - min + 1e-8);
        const cm = COLORMAPS[selectedColormap] || COLORMAPS.turbo;
        return cm(t);
    }

    function _detectMobile() {
        const ua = navigator.userAgent || navigator.vendor || '';
        const mobileUA = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile/i.test(ua);
        const smallScreen = window.innerWidth < 768 || window.innerHeight < 600;
        const lowCores = (navigator.hardwareConcurrency || 8) <= 4;
        isMobileFlag = mobileUA || smallScreen || lowCores;
        return isMobileFlag;
    }

    function showToast(title, message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
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

    function drawHeatmap(thicknessData) {
        const canvas = document.getElementById('heatmap-canvas');
        if (!canvas || !thicknessData) return;

        const { thickness_um, min_um, max_um } = thicknessData;
        const grid = thickness_um.length;
        const gs = grid || 48;

        const rect = canvas.parentElement.getBoundingClientRect();
        const size = Math.min(rect.width - 24, isMobileFlag ? 250 : 350);
        if (canvas.width !== size) {
            canvas.width = size;
            canvas.height = size;
        }

        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(size, size);
        const data = imgData.data;

        for (let py = 0; py < size; py++) {
            const si = (py / size) * gs;
            const i0 = Math.min(Math.floor(si), gs - 1);
            const i1 = Math.min(i0 + 1, gs - 1);
            const fi = si - i0;
            for (let px = 0; px < size; px++) {
                const sj = (px / size) * gs;
                const j0 = Math.min(Math.floor(sj), gs - 1);
                const j1 = Math.min(j0 + 1, gs - 1);
                const fj = sj - j0;
                const v00 = thickness_um[i0][j0];
                const v10 = thickness_um[i1][j0];
                const v01 = thickness_um[i0][j1];
                const v11 = thickness_um[i1][j1];
                const v = v00 * (1 - fi) * (1 - fj)
                    + v10 * fi * (1 - fj)
                    + v01 * (1 - fi) * fj
                    + v11 * fi * fj;
                const rgb = _getColormapColor(v, min_um, max_um);
                const idx = (py * size + px) * 4;
                data[idx] = rgb[0];
                data[idx + 1] = rgb[1];
                data[idx + 2] = rgb[2];
                data[idx + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);

        ctx.strokeStyle = 'rgba(212, 175, 55, 0.35)';
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
        currentThicknessData = thicknessData;
    }

    function _setupCanvasTooltip() {
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

    function updateMetrics(state) {
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

        if (metrics.grid_size !== undefined) {
            document.getElementById('metric-gridsize').textContent =
                metrics.grid_size + '×' + metrics.grid_size;
        } else if (state.grid_size !== undefined) {
            document.getElementById('metric-gridsize').textContent =
                state.grid_size + '×' + state.grid_size;
        }
    }

    function updateRisk(risk) {
        const indicator = document.getElementById('risk-indicator');
        if (!indicator) return;
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

    function addAlert(alert) {
        if (alertsData.length === 0) {
            document.querySelector('#alerts-log .log-empty')?.remove();
        }
        alertsData.unshift(alert);
        if (alertsData.length > 50) alertsData.pop();

        const log = document.getElementById('alerts-log');
        if (!log) return;
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

    function addStrikeHistory(result) {
        const strike = result.strike || result;
        if (strikeHistoryData.length === 0) {
            document.querySelector('#strike-history .log-empty')?.remove();
        }
        strikeHistoryData.unshift(strike);
        if (strikeHistoryData.length > 100) strikeHistoryData.pop();

        const log = document.getElementById('strike-history');
        if (!log) return;
        const item = document.createElement('div');
        item.className = 'strike-item';

        const num = strike.strike_num || strikeHistoryData.length;
        const force = strike.hammer_force_N || result.action?.force_N || 0;
        const thick = strike.avg_thickness_um || strike.metrics?.mean_thickness_um || 0;
        const pos = strike.hammer_position || result.action?.position_mm || [0, 0];
        const remesh = strike.remesh || result.remesh || null;
        const remeshTag = remesh && remesh.action && remesh.action !== 'noop'
            ? `<span style="color:#00e0ff;font-size:10px;">[${remesh.action} ${remesh.old_size}→${remesh.new_size}]</span>`
            : '';

        item.innerHTML = `
            <span class="strike-num">#${num}</span>
            <span class="strike-info">${force.toFixed(0)}N<br>(${pos[0].toFixed(0)},${pos[1].toFixed(0)}) ${remeshTag}</span>
            <span class="strike-thickness">${thick.toFixed(2)}μm</span>
        `;
        log.insertBefore(item, log.firstChild);
    }

    function setEventCallbacks(cb) {
        callbacks = { ...callbacks, ...cb };
    }

    function _setupEventListeners() {
        const btn = document.getElementById('btn-strike');
        if (btn) btn.addEventListener('click', () => callbacks.onStrike?.());

        document.getElementById('btn-auto')?.addEventListener('click',
            () => callbacks.onAutoStart?.());
        document.getElementById('btn-stop')?.addEventListener('click',
            () => callbacks.onAutoStop?.());
        document.getElementById('btn-anneal')?.addEventListener('click',
            () => callbacks.onAnneal?.());
        document.getElementById('btn-reset')?.addEventListener('click',
            () => callbacks.onReset?.());
        document.getElementById('btn-export')?.addEventListener('click',
            () => callbacks.onExportJSON?.());
        document.getElementById('btn-export-csv')?.addEventListener('click',
            () => callbacks.onExportCSV?.());

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

        document.getElementById('strike-mode')?.addEventListener('change', (e) => {
            const manual = document.getElementById('manual-controls');
            if (manual) {
                manual.style.display = e.target.value === 'manual' ? 'block' : 'none';
            }
        });

        document.getElementById('colormap-select')?.addEventListener('change', (e) => {
            selectedColormap = e.target.value;
            callbacks.onColormapChange?.(selectedColormap);
            if (currentThicknessData) {
                drawHeatmap(currentThicknessData);
            }
        });

        ['toggle-wireframe', 'toggle-color', 'toggle-auto-rotate'].forEach(id => {
            document.getElementById(id)?.addEventListener('change', () => {
                const wire = document.getElementById('toggle-wireframe')?.checked;
                const color = document.getElementById('toggle-color')?.checked;
                const rotate = document.getElementById('toggle-auto-rotate')?.checked;
                callbacks.onDisplayChange?.({ wireframe: wire, color: color, autoRotate: rotate });
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.code === 'Space') { e.preventDefault(); callbacks.onStrike?.(); }
            if (e.code === 'KeyR') callbacks.onReset?.();
            if (e.code === 'KeyA') {
                if (document.getElementById('btn-stop')?.style.display === 'flex') {
                    callbacks.onAutoStop?.();
                } else {
                    callbacks.onAutoStart?.();
                }
            }
        });
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

    function getStrikeMode() {
        return document.getElementById('strike-mode')?.value || 'heuristic';
    }

    function getManualParams() {
        return {
            force_N: parseFloat(document.getElementById('force-slider')?.value || 500),
            position_x_mm: parseFloat(document.getElementById('posx-slider')?.value || 0),
            position_y_mm: parseFloat(document.getElementById('posy-slider')?.value || 0),
            radius_mm: 15,
        };
    }

    function getAutoInterval() {
        return parseFloat(document.getElementById('interval-slider')?.value || 1);
    }

    function getAnnealTemp() {
        return parseFloat(document.getElementById('anneal-temp')?.value || 400);
    }

    function getColormap() {
        return selectedColormap;
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

    function clearHistory() {
        strikeHistoryData = [];
        alertsData = [];
        document.getElementById('strike-history').innerHTML =
            '<div class="log-empty">等待锤击...</div>';
        document.getElementById('alerts-log').innerHTML =
            '<div class="log-empty">暂无告警</div>';
    }

    function init(options = {}) {
        foilSize = options.foilSize || 150;
        _detectMobile();
        _setupEventListeners();
        _setupCanvasTooltip();
        console.log('[ThicknessPanel] 初始化完成');
    }

    return {
        init,
        updateMetrics,
        updateRisk,
        updateHeatmap: drawHeatmap,
        addAlert,
        addStrikeHistory,
        setEventCallbacks,
        fetchJSON,
        showToast,
        exportJSON,
        exportCSV,
        clearHistory,
        getStrikeMode,
        getManualParams,
        getAutoInterval,
        getAnnealTemp,
        getColormap,
        get currentThicknessData() { return currentThicknessData; },
        get strikeHistory() { return strikeHistoryData; },
        get alerts() { return alertsData; },
        get isMobile() { return isMobileFlag; },
    };
})();
