/**
 * GoldFoil3D - 金箔三维渲染模块 (GPU Shader优化版)
 *
 * 职责:
 *   - Three.js 场景 / 相机 / 渲染器 / 光照
 *   - 金箔 ShaderMaterial + DataTexture 厚度渲染
 *   - 锤头模型与锤击动画
 *   - 色图 LUT 切换
 *   - OrbitControls 交互
 *
 * 对外 API:
 *   GoldFoil3D.init(containerId, options)
 *   GoldFoil3D.updateThickness(thicknessData)
 *   GoldFoil3D.animateStrike(position, force)
 *   GoldFoil3D.setColormap(name)
 *   GoldFoil3D.setWireframe(enabled)
 *   GoldFoil3D.setColorEnabled(enabled)
 *   GoldFoil3D.setAutoRotate(enabled)
 *   GoldFoil3D.resize()
 *   GoldFoil3D.isMobile
 *   GoldFoil3D.foilSize
 */

const GoldFoil3D = (function () {
    "use strict";

    let scene, camera, renderer, controls, foilMesh, hammerMesh;
    let foilBaseGeometry = null;
    let foilShaderMaterial = null;
    let thicknessDataTexture = null;
    let colormapTexture = null;
    let animationFrameId = null;
    let isMobileFlag = false;
    let vizThicknessRange = { min: 0, max: 500 };
    let renderGridSize = 48;
    let foilSizeMm = 150;

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

    const FOIL_VERTEX_SHADER = `
        uniform sampler2D uThicknessTexture;
        uniform float uThicknessMin;
        uniform float uThicknessMax;
        uniform float uHeightScale;
        uniform float uBaseHeight;
        uniform float uTime;
        uniform float uHammerIntensity;
        uniform vec2  uHammerPosition;
        uniform float uHammerRadius;

        varying vec2 vUv;
        varying float vThicknessNormalized;
        varying float vWorldHeight;

        void main() {
            vUv = uv;

            vec4 texel = texture2D(uThicknessTexture, uv);
            float thickness = texel.r;
            float range = max(uThicknessMax - uThicknessMin, 0.0001);
            vThicknessNormalized = clamp((thickness - uThicknessMin) / range, 0.0, 1.0);

            float baseY = position.y;
            float heightOffset = vThicknessNormalized * uHeightScale;

            float dx = (position.x + uHammerPosition.x);
            float dz = (position.z + uHammerPosition.y);
            float dist2 = dx * dx + dz * dz;
            float hammerDisturb = 0.0;
            if (uHammerIntensity > 0.001 && dist2 < uHammerRadius * uHammerRadius) {
                float d = sqrt(dist2) / uHammerRadius;
                float gauss = exp(-d * d * 4.0);
                hammerDisturb = -gauss * uHammerIntensity * 2.0;
            }

            float finalY = uBaseHeight + heightOffset + hammerDisturb;
            vWorldHeight = finalY;

            vec3 newPosition = vec3(position.x, finalY, position.z);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
    `;

    const FOIL_FRAGMENT_SHADER = `
        uniform sampler2D uColormapLUT;
        uniform float uShininess;
        uniform int   uUseColor;
        uniform float uTime;
        uniform vec3  uGoldColor;

        varying vec2 vUv;
        varying float vThicknessNormalized;
        varying float vWorldHeight;

        void main() {
            vec3 lutColor;
            if (uUseColor == 1) {
                vec4 tex = texture2D(uColormapLUT, vec2(vThicknessNormalized, 0.5));
                lutColor = tex.rgb;
            } else {
                lutColor = uGoldColor;
            }

            float df = fwidth(vThicknessNormalized) * 2.0;
            float grad_light = 0.6 + 0.4 * vThicknessNormalized;
            vec3 finalColor = lutColor * grad_light;

            float fresnel = pow(1.0 - max(dot(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)), 0.0), 2.0);
            finalColor += fresnel * vec3(0.2, 0.18, 0.1);

            gl_FragColor = vec4(finalColor, 1.0);
        }
    `;

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

    function _detectMobile() {
        const ua = navigator.userAgent || navigator.vendor || '';
        const mobileUA = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile/i.test(ua);
        const smallScreen = window.innerWidth < 768 || window.innerHeight < 600;
        const lowCores = (navigator.hardwareConcurrency || 8) <= 4;
        isMobileFlag = mobileUA || smallScreen || lowCores;
        return isMobileFlag;
    }

    function _buildColormapLUT(name) {
        const fn = COLORMAPS[name] || COLORMAPS.turbo;
        const data = new Uint8Array(COLORMAP_LUT_SIZE * 4);
        for (let i = 0; i < COLORMAP_LUT_SIZE; i++) {
            const t = i / (COLORMAP_LUT_SIZE - 1);
            const rgb = fn(t);
            data[i * 4 + 0] = rgb[0];
            data[i * 4 + 1] = rgb[1];
            data[i * 4 + 2] = rgb[2];
            data[i * 4 + 3] = 255;
        }
        return data;
    }

    function _createColormapTexture(name) {
        const lutData = _buildColormapLUT(name);
        const tex = new THREE.DataTexture(
            lutData,
            COLORMAP_LUT_SIZE,
            1,
            THREE.RGBAFormat,
            THREE.UnsignedByteType
        );
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.needsUpdate = true;
        return tex;
    }

    function _createFoilMesh() {
        renderGridSize = isMobileFlag ? 32 : (64 > 64 ? 64 : 48);

        foilBaseGeometry = new THREE.PlaneGeometry(
            foilSizeMm,
            foilSizeMm,
            renderGridSize - 1,
            renderGridSize - 1
        );
        foilBaseGeometry.rotateX(-Math.PI / 2);

        colormapTexture = _createColormapTexture('turbo');

        const initialData = new Float32Array(renderGridSize * renderGridSize);
        initialData.fill(500.0);
        thicknessDataTexture = new THREE.DataTexture(
            initialData,
            renderGridSize,
            renderGridSize,
            THREE.RedFormat,
            THREE.FloatType
        );
        thicknessDataTexture.wrapS = THREE.ClampToEdgeWrapping;
        thicknessDataTexture.wrapT = THREE.ClampToEdgeWrapping;
        thicknessDataTexture.minFilter = THREE.LinearFilter;
        thicknessDataTexture.magFilter = THREE.LinearFilter;
        thicknessDataTexture.needsUpdate = true;

        foilShaderMaterial = new THREE.ShaderMaterial({
            uniforms: {
                uThicknessTexture: { value: thicknessDataTexture },
                uThicknessMin: { value: 0.0 },
                uThicknessMax: { value: 500.0 },
                uHeightScale: { value: 4.0 },
                uBaseHeight: { value: -2.0 },
                uColormapLUT: { value: colormapTexture },
                uShininess: { value: 100.0 },
                uUseColor: { value: 1 },
                uTime: { value: 0 },
                uGoldColor: { value: new THREE.Color(0xd4af37) },
                uHammerIntensity: { value: 0.0 },
                uHammerPosition: { value: new THREE.Vector2(0, 0) },
                uHammerRadius: { value: 30.0 },
            },
            vertexShader: FOIL_VERTEX_SHADER,
            fragmentShader: FOIL_FRAGMENT_SHADER,
            side: THREE.DoubleSide,
            wireframe: false,
        });

        foilMesh = new THREE.Mesh(foilBaseGeometry, foilShaderMaterial);
        if (!isMobileFlag) {
            foilMesh.receiveShadow = true;
            foilMesh.castShadow = true;
        }
        scene.add(foilMesh);
    }

    function _createHammerMesh() {
        const handleGeo = new THREE.CylinderGeometry(2, 2, 60, isMobileFlag ? 8 : 16);
        const handleMat = new THREE.MeshPhongMaterial({ color: 0x8B4513, shininess: 20 });
        const handle = new THREE.Mesh(handleGeo, handleMat);

        const headGeo = new THREE.CylinderGeometry(10, 10, 20, isMobileFlag ? 8 : 16);
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

    function _setupLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, isMobileFlag ? 0.6 : 0.5);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, isMobileFlag ? 0.8 : 1.0);
        dirLight.position.set(100, 200, 100);
        if (!isMobileFlag) {
            dirLight.castShadow = true;
            dirLight.shadow.mapSize.set(512, 512);
        }
        scene.add(dirLight);

        if (!isMobileFlag) {
            const pointLight = new THREE.PointLight(0xffd700, 0.6, 500);
            pointLight.position.set(-50, 80, -50);
            scene.add(pointLight);
        }
    }

    function _animate() {
        animationFrameId = requestAnimationFrame(_animate);
        if (controls && document.getElementById('toggle-auto-rotate')?.checked) {
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.5;
        } else if (controls) {
            controls.autoRotate = false;
        }
        if (controls) controls.update();
        if (foilShaderMaterial) {
            foilShaderMaterial.uniforms.uTime.value = performance.now() * 0.001;
        }
        if (renderer && scene && camera) renderer.render(scene, camera);
    }

    function _onResize() {
        const container = document.getElementById('three-container');
        if (!container || !camera || !renderer) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function init(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('[GoldFoil3D] Container not found:', containerId);
            return false;
        }

        foilSizeMm = options.foilSize || 150;
        _detectMobile();

        const width = container.clientWidth;
        const height = container.clientHeight;

        scene = new THREE.Scene();
        scene.background = null;

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
        camera.position.set(0, 180, 200);

        renderer = new THREE.WebGLRenderer({
            antialias: !isMobileFlag,
            alpha: true,
            powerPreference: isMobileFlag ? 'low-power' : 'high-performance'
        });
        renderer.setSize(width, height);
        const dpr = isMobileFlag ? Math.min(window.devicePixelRatio, 1.5) : window.devicePixelRatio;
        renderer.setPixelRatio(dpr);
        renderer.shadowMap.enabled = !isMobileFlag;
        if (!isMobileFlag) renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = isMobileFlag ? 0.15 : 0.08;
        controls.target.set(0, 0, 0);
        controls.enablePan = !isMobileFlag;

        _setupLights();
        _createFoilMesh();
        _createHammerMesh();

        const gridHelper = new THREE.GridHelper(
            foilSizeMm * 1.5,
            isMobileFlag ? 10 : 20,
            0x333333,
            0x222222
        );
        gridHelper.position.y = -2.5;
        scene.add(gridHelper);

        window.addEventListener('resize', _onResize);
        _animate();

        console.log('[GoldFoil3D] 初始化完成, 模式:', isMobileFlag ? '移动端' : '桌面端', `(${renderGridSize}x${renderGridSize})`);
        return true;
    }

    function updateThickness(thicknessData) {
        if (!foilMesh || !foilShaderMaterial || !thicknessData) return;

        const { thickness_um, min_um, max_um, grid_size } = thicknessData;
        const srcGrid = thickness_um.length;

        vizThicknessRange.min = min_um;
        vizThicknessRange.max = max_um;
        foilShaderMaterial.uniforms.uThicknessMin.value = min_um;
        foilShaderMaterial.uniforms.uThicknessMax.value = max_um;

        const tex = foilShaderMaterial.uniforms.uThicknessTexture.value;
        const dstGrid = tex.image.width;
        const dstData = tex.image.data;

        if (srcGrid === dstGrid) {
            for (let i = 0; i < srcGrid; i++) {
                for (let j = 0; j < srcGrid; j++) {
                    dstData[i * dstGrid + j] = thickness_um[i][j];
                }
            }
        } else {
            const srcArr = thickness_um;
            for (let di = 0; di < dstGrid; di++) {
                for (let dj = 0; dj < dstGrid; dj++) {
                    const si = (di / (dstGrid - 1)) * (srcGrid - 1);
                    const sj = (dj / (dstGrid - 1)) * (srcGrid - 1);
                    const i0 = Math.floor(si);
                    const j0 = Math.floor(sj);
                    const i1 = Math.min(i0 + 1, srcGrid - 1);
                    const j1 = Math.min(j0 + 1, srcGrid - 1);
                    const fi = si - i0;
                    const fj = sj - j0;
                    const v00 = srcArr[i0][j0];
                    const v10 = srcArr[i1][j0];
                    const v01 = srcArr[i0][j1];
                    const v11 = srcArr[i1][j1];
                    const v = v00 * (1 - fi) * (1 - fj)
                        + v10 * fi * (1 - fj)
                        + v01 * (1 - fi) * fj
                        + v11 * fi * fj;
                    dstData[di * dstGrid + dj] = v;
                }
            }
        }

        tex.needsUpdate = true;

        document.getElementById('legend-min').textContent = min_um.toFixed(2);
        document.getElementById('legend-max').textContent = max_um.toFixed(2);
    }

    function animateStrike(position, force) {
        if (!hammerMesh) return;

        const startPos = { y: 80, rx: Math.PI / 6 };
        const endPos = { y: 10, rx: 0 };
        const duration = isMobileFlag ? 250 : 300;
        const startTime = performance.now();

        hammerMesh.position.x = position[0];
        hammerMesh.position.z = position[1];

        const normalizedForce = Math.min(Math.max((force - 300) / 1200, 0), 1);
        if (foilShaderMaterial) {
            foilShaderMaterial.uniforms.uHammerPosition.value.set(-position[0], -position[1]);
            foilShaderMaterial.uniforms.uHammerRadius.value = 30.0;
        }

        function tick() {
            const elapsed = performance.now() - startTime;
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

            if (foilShaderMaterial) {
                foilShaderMaterial.uniforms.uHammerIntensity.value = eased * normalizedForce;
            }

            if (progress < 1) {
                requestAnimationFrame(tick);
            } else {
                if (foilShaderMaterial) {
                    setTimeout(() => {
                        if (foilShaderMaterial) foilShaderMaterial.uniforms.uHammerIntensity.value = 0;
                    }, 80);
                }
                setTimeout(() => {
                    hammerMesh.position.set(0, 80, 0);
                    hammerMesh.rotation.z = Math.PI / 6;
                }, 100);
            }
        }
        tick();
    }

    function setColormap(name) {
        if (!foilShaderMaterial) return;
        if (foilShaderMaterial.uniforms.uColormapLUT.value) {
            foilShaderMaterial.uniforms.uColormapLUT.value.dispose();
        }
        colormapTexture = _createColormapTexture(name);
        foilShaderMaterial.uniforms.uColormapLUT.value = colormapTexture;
    }

    function setWireframe(enabled) {
        if (foilShaderMaterial) foilShaderMaterial.wireframe = enabled;
    }

    function setColorEnabled(enabled) {
        if (foilShaderMaterial) {
            foilShaderMaterial.uniforms.uUseColor.value = enabled ? 1 : 0;
        }
    }

    function setAutoRotate(enabled) {
        if (controls) controls.autoRotate = enabled;
    }

    function resize() {
        _onResize();
    }

    function getColormaps() {
        return Object.keys(COLORMAPS);
    }

    function getThicknessRange() {
        return { ...vizThicknessRange };
    }

    return {
        init,
        updateThickness,
        animateStrike,
        setColormap,
        setWireframe,
        setColorEnabled,
        setAutoRotate,
        resize,
        getColormaps,
        getThicknessRange,
        get isMobile() { return isMobileFlag; },
        get foilSize() { return foilSizeMm; },
    };
})();
