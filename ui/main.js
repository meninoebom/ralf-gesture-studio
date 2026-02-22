// RALF Gesture Studio - Frontend JavaScript

const { invoke } = window.__TAURI__.core;

// Application State
let state = {
    mode: 'training',
    vocabulary: null,
    selectedGestureId: null,
    trainingState: 'idle',
    dirty: false,
    isEditingGestureName: false, // Prevent re-render during inline editing
    lastGesturesHash: null, // Track changes to avoid unnecessary re-renders
    lastTrainingHash: null, // Track training state changes
    diagnosticsEnabled: false,
    diagnosticsPath: null,
    expandedGestures: new Set(), // Track which gesture example lists are expanded
};

// DOM Elements
const elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    cacheElements();
    setupEventListeners();
    await loadInitialState();
    startPolling();
});

function cacheElements() {
    elements.header = document.getElementById('header');
    elements.dirtyIndicator = document.getElementById('dirty-indicator');
    elements.btnTraining = document.getElementById('btn-training');
    elements.btnPerformance = document.getElementById('btn-performance');

    elements.vocabName = document.getElementById('vocab-name');
    elements.vocabPath = document.getElementById('vocab-path');
    elements.gestureCount = document.getElementById('gesture-count');
    elements.btnNew = document.getElementById('btn-new');
    elements.btnOpen = document.getElementById('btn-open');
    elements.btnSave = document.getElementById('btn-save');

    elements.connectionStatus = document.getElementById('connection-status');
    elements.inputStatus = document.getElementById('input-status');
    elements.inputStatusDot = document.getElementById('input-status-dot');
    elements.frameCount = document.getElementById('frame-count');
    elements.outputStatus = document.getElementById('output-status');
    elements.outputStatusDot = document.getElementById('output-status-dot');
    elements.sendCount = document.getElementById('send-count');
    elements.btnTestHit = document.getElementById('btn-test-hit');

    // Popover elements
    elements.connectionPopover = document.getElementById('connection-popover');
    elements.popoverInputPort = document.getElementById('popover-input-port');
    elements.popoverInputAddress = document.getElementById('popover-input-address');
    elements.popoverInputTime = document.getElementById('popover-input-time');
    elements.popoverOutputHost = document.getElementById('popover-output-host');
    elements.popoverOutputPort = document.getElementById('popover-output-port');
    elements.popoverOutputTime = document.getElementById('popover-output-time');
    elements.dimensionMismatch = document.getElementById('dimension-mismatch');
    elements.popoverDimDetail = document.getElementById('popover-dim-detail');

    elements.gestureList = document.getElementById('gesture-list');
    elements.btnAddGesture = document.getElementById('btn-add-gesture');

    elements.trainingPanel = document.getElementById('training-panel');
    elements.selectedGesture = document.getElementById('selected-gesture');
    elements.trainReps = document.getElementById('train-reps');
    elements.trainCountdown = document.getElementById('train-countdown');
    elements.trainDuration = document.getElementById('train-duration');
    elements.trainRest = document.getElementById('train-rest');
    elements.trainingDisplay = document.getElementById('training-display');
    elements.btnStartTraining = document.getElementById('btn-start-training');
    elements.trainingHint = document.getElementById('training-hint');
    elements.trainingStatus = document.getElementById('training-status');
    elements.qualityFeedback = document.getElementById('quality-feedback');
    elements.btnAugmentation = document.getElementById('btn-augmentation');
    elements.btnJointWeighting = document.getElementById('btn-joint-weighting');
    elements.btnComplexityCorrection = document.getElementById('btn-complexity-correction');
    elements.btnF1Threshold = document.getElementById('btn-f1-threshold');
    elements.confusionWarning = document.getElementById('confusion-warning');

    elements.performancePanel = document.getElementById('performance-panel');
    elements.hitDisplay = document.getElementById('hit-display');
    elements.recognizerStatus = document.getElementById('recognizer-status');
    elements.bufferStatus = document.getElementById('buffer-status');
    elements.windowStatus = document.getElementById('window-status');
    elements.exampleStatus = document.getElementById('example-status');
    elements.monitorList = document.getElementById('monitor-list');
    elements.cooldownMs = document.getElementById('cooldown-ms');
    elements.hitLog = document.getElementById('hit-log');
    elements.hitTotal = document.getElementById('hit-total');
    elements.btnToggleDiagnostics = document.getElementById('btn-toggle-diagnostics');
    elements.diagnosticsStatus = document.getElementById('diagnostics-status');
}

function setupEventListeners() {
    // Mode toggle
    elements.btnTraining.addEventListener('click', () => setMode('training'));
    elements.btnPerformance.addEventListener('click', () => setMode('performance'));

    // Vocabulary actions
    elements.btnNew.addEventListener('click', newVocabulary);
    elements.btnOpen.addEventListener('click', openVocabulary);
    elements.btnSave.addEventListener('click', saveVocabulary);

    // Connection
    elements.btnTestHit.addEventListener('click', (e) => {
        e.stopPropagation(); // Don't trigger popover
        sendTestHit();
    });

    // Connection popover toggle
    elements.connectionStatus.addEventListener('click', (e) => {
        // Don't toggle if clicking the test button
        if (e.target.id === 'btn-test-hit') return;
        toggleConnectionPopover();
    });

    // Close popover when clicking outside
    document.addEventListener('click', (e) => {
        if (!elements.connectionStatus.contains(e.target)) {
            elements.connectionPopover.classList.add('hidden');
        }
    });

    // Gestures
    elements.btnAddGesture.addEventListener('click', addGesture);

    // Training start button — click delegation (idle state doesn't re-render, so click is fine)
    elements.trainingDisplay.addEventListener('click', (e) => {
        if (e.target.closest('#btn-start-training')) {
            startTraining();
        }
    });

    // Stop training — mousedown on document fires immediately on press,
    // before the 50ms poll can destroy the button via innerHTML replacement.
    document.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        if (e.target.closest('.stop-training-btn')) {
            e.preventDefault();
            cancelTraining();
        }
    });

    // Keyboard shortcuts (skip when an input/textarea is focused)
    document.addEventListener('keydown', (e) => {
        const tag = document.activeElement?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;

        if (e.code === 'Space' && state.mode === 'training' && state.trainingState === 'idle') {
            e.preventDefault();
            startTraining();
        }
        if (e.code === 'Escape') {
            cancelTraining();
        }
    });

    // Cooldown change
    elements.cooldownMs.addEventListener('change', async () => {
        await invoke('set_cooldown', { ms: parseInt(elements.cooldownMs.value) });
    });

    // Augmentation toggle
    elements.btnAugmentation.addEventListener('click', toggleAugmentation);

    // Joint weighting toggle
    elements.btnJointWeighting.addEventListener('click', toggleJointWeighting);
    elements.btnComplexityCorrection.addEventListener('click', toggleComplexityCorrection);
    elements.btnF1Threshold.addEventListener('click', toggleF1Threshold);

    // Diagnostics toggle
    elements.btnToggleDiagnostics.addEventListener('click', toggleDiagnostics);
}

async function loadInitialState() {
    try {
        const appState = await invoke('get_state');
        updateFromState(appState);
    } catch (e) {
        console.error('Failed to load initial state:', e);
    }
}

function startPolling() {
    // Poll for state updates every 50ms
    setInterval(async () => {
        try {
            const appState = await invoke('get_state');
            updateFromState(appState);
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 50);
}

function updateFromState(appState) {
    // Selected gesture ID from backend
    state.selectedGestureId = appState.selected_gesture_id;

    // Vocabulary
    if (appState.vocabulary) {
        state.vocabulary = appState.vocabulary;
        elements.vocabName.textContent = appState.vocabulary.name;
        elements.gestureCount.textContent = `Gestures: ${appState.vocabulary.gestures.length}`;
        renderGestures(appState.vocabulary.gestures);
    }

    // File path
    elements.vocabPath.textContent = appState.file_path ? `(${appState.file_path})` : '';

    // Dirty indicator
    state.dirty = appState.dirty;
    elements.dirtyIndicator.classList.toggle('hidden', !appState.dirty);

    // Connection status
    updateConnectionStatus(appState.osc_status);

    // Augmentation toggle state
    if (appState.vocabulary?.augmentation) {
        const aug = appState.vocabulary.augmentation;
        elements.btnAugmentation.textContent = aug.enabled ? `ON (×${aug.multiplier})` : 'OFF';
        elements.btnAugmentation.classList.toggle('active', aug.enabled);
    }

    // Joint weighting toggle state
    if (appState.vocabulary) {
        const jw = appState.vocabulary.joint_weighting;
        elements.btnJointWeighting.textContent = jw ? 'ON' : 'OFF';
        elements.btnJointWeighting.classList.toggle('active', jw);

        const cc = appState.vocabulary.complexity_correction;
        elements.btnComplexityCorrection.textContent = cc ? 'ON' : 'OFF';
        elements.btnComplexityCorrection.classList.toggle('active', cc);

        const f1 = appState.vocabulary.f1_threshold;
        elements.btnF1Threshold.textContent = f1 ? 'ON' : 'OFF';
        elements.btnF1Threshold.classList.toggle('active', f1);

        // Confusion warnings
        const pairs = appState.vocabulary.confusion_pairs || [];
        if (pairs.length > 0) {
            elements.confusionWarning.classList.remove('hidden');
            elements.confusionWarning.innerHTML = pairs.map(p =>
                `<span class="confusion-item">⚠ "${p.gesture_name_a}" and "${p.gesture_name_b}" may be confused (${Math.round(p.overlap_ratio * 100)}% overlap)</span>`
            ).join('');
        } else {
            elements.confusionWarning.classList.add('hidden');
            elements.confusionWarning.innerHTML = '';
        }
    }

    // Training state
    updateTrainingState(appState.training);

    // Performance mode data
    if (state.mode === 'performance') {
        updateMonitor(appState.monitor);
        updateHitLog(appState.hit_log);
    }
}

function toggleConnectionPopover() {
    elements.connectionPopover.classList.toggle('hidden');
}

function formatTimeAgo(ms) {
    if (ms === 0 || ms === undefined) return '—';
    if (ms < 1000) return `${ms}ms ago`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s ago`;
    return `${Math.floor(ms / 60000)}m ago`;
}

function updateConnectionStatus(status) {
    if (!status) return;

    // Input
    const inputColors = {
        'Stopped': ['gray', 'STOPPED'],
        'Listening': ['gold', 'LISTENING'],
        'Receiving': ['green', 'RECEIVING'],
        'Error': ['red', 'ERROR'],
    };

    let [dotClass, statusText] = inputColors[status.input_status] || ['gray', 'UNKNOWN'];

    // Override with dimension mismatch warning
    if (status.dimension_mismatch_expected != null) {
        dotClass = 'red';
        statusText = 'DIM MISMATCH';
        elements.dimensionMismatch.classList.remove('hidden');
        elements.popoverDimDetail.textContent =
            `Expected ${status.dimension_mismatch_expected}, receiving ${status.dimension_mismatch_actual}`;
    } else {
        elements.dimensionMismatch.classList.add('hidden');
    }

    elements.inputStatusDot.className = `status-dot ${dotClass}`;
    elements.inputStatus.textContent = statusText;
    elements.inputStatus.className = dotClass;
    elements.frameCount.textContent = status.frame_count;

    // Output
    const outputColors = {
        'Ready': ['green', 'READY'],
        'Sent': ['green', 'SENT'],
        'Error': ['red', 'ERROR'],
    };

    const [outDot, outStatus] = outputColors[status.output_status] || ['green', 'READY'];
    elements.outputStatusDot.className = `status-dot ${outDot}`;
    elements.outputStatus.textContent = outStatus;
    elements.outputStatus.className = outDot;
    elements.sendCount.textContent = status.send_count;

    // Update popover with detailed timing info
    elements.popoverInputTime.textContent = formatTimeAgo(status.ms_since_last_frame);
    elements.popoverOutputTime.textContent = formatTimeAgo(status.ms_since_last_send);
}

function buildExampleList(gesture) {
    const list = document.createElement('div');
    list.className = 'example-list';

    gesture.examples.forEach((ex, idx) => {
        const time = new Date(ex.recorded_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const duration = (ex.duration_ms / 1000).toFixed(1);
        const isOutlier = gesture.outlier_example_indices && gesture.outlier_example_indices.includes(idx);
        const isMedoid = gesture.medoid_index === idx;

        const row = document.createElement('div');
        row.className = 'example-item' + (isOutlier ? ' outlier' : '') + (isMedoid ? ' medoid' : '');

        const info = document.createElement('span');
        info.className = 'example-info dim';
        info.textContent = `${isMedoid ? '★ ' : ''}${time} · ${duration}s · ${ex.frame_count} frames${isOutlier ? ' ⚠ outlier' : ''}`;

        const btn = document.createElement('button');
        btn.className = 'example-delete-btn';
        btn.textContent = '×';
        btn.onclick = () => deleteExample(gesture.id, idx);

        row.appendChild(info);
        row.appendChild(btn);
        list.appendChild(row);
    });

    return list;
}

function consistencyBadge(consistency) {
    if (consistency == null) return '';
    if (consistency < 0.3) return ' <span class="consistency-badge good" title="Consistent examples">●</span>';
    if (consistency < 0.6) return ' <span class="consistency-badge fair" title="Moderate variance">●</span>';
    return ' <span class="consistency-badge poor" title="High variance — consider removing outliers">●</span>';
}

function renderGestures(gestures) {
    // Skip re-render if currently editing a gesture name
    if (state.isEditingGestureName) {
        // Still update the training panel's selected gesture display
        const selected = gestures.find(g => g.id === state.selectedGestureId);
        elements.selectedGesture.textContent = selected ? selected.name : '(none)';
        return;
    }

    // Create a hash of the gesture data + selected ID + expanded state to detect changes
    const currentHash = JSON.stringify({
        gestures: gestures.map(g => ({ id: g.id, name: g.name, examples: g.examples.length, osc_address: g.osc_address })),
        selectedId: state.selectedGestureId,
        expanded: [...state.expandedGestures],
    });

    // Skip re-render if nothing changed
    if (currentHash === state.lastGesturesHash) {
        return;
    }
    state.lastGesturesHash = currentHash;

    elements.gestureList.innerHTML = '';

    for (const gesture of gestures) {
        const isExpanded = state.expandedGestures.has(gesture.id);
        const container = document.createElement('div');
        container.className = 'gesture-container';

        const row = document.createElement('div');
        row.className = 'gesture-row';
        row.innerHTML = `
            <span class="col-status">
                <span class="status-dot ${gesture.examples.length > 0 ? 'green' : 'gray'}">
                    ${gesture.examples.length > 0 ? '●' : '○'}
                </span>
            </span>
            <span class="name col-name ${state.selectedGestureId === gesture.id ? 'selected' : ''}"
                  data-id="${gesture.id}">${gesture.name}${consistencyBadge(gesture.consistency)}</span>
            <span class="examples col-examples clickable ${isExpanded ? 'expanded' : ''}"
                  data-id="${gesture.id}"
                  title="${gesture.examples.length > 0 ? 'Click to expand examples' : ''}">${gesture.examples.length}${gesture.examples.length > 0 ? (isExpanded ? ' ▾' : ' ▸') : ''}</span>
            <span class="address col-address">${gesture.osc_address}</span>
            <span class="actions col-actions">
                <button class="btn-select" data-id="${gesture.id}">
                    ${state.selectedGestureId === gesture.id ? 'Selected' : 'Select'}
                </button>
                <button class="btn-delete" data-id="${gesture.id}">×</button>
            </span>
        `;

        const nameEl = row.querySelector('.name');
        nameEl.addEventListener('click', (e) => {
            e.stopPropagation();
            renameGesture(gesture.id, nameEl);
        });

        // Toggle example list on example count click
        const examplesEl = row.querySelector('.examples');
        if (gesture.examples.length > 0) {
            examplesEl.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleExampleList(gesture.id);
            });
        }

        row.querySelector('.btn-select').addEventListener('click', (e) => {
            e.stopPropagation();
            selectGesture(gesture.id);
        });
        row.querySelector('.btn-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteGesture(gesture.id);
        });

        container.appendChild(row);

        // Render expanded example list
        if (isExpanded && gesture.examples.length > 0) {
            container.appendChild(buildExampleList(gesture));
        }

        elements.gestureList.appendChild(container);
    }

    // Update selected gesture name in training panel
    const selected = gestures.find(g => g.id === state.selectedGestureId);
    elements.selectedGesture.textContent = selected ? selected.name : '(none)';
}

function updateTrainingState(training) {
    if (!training) return;

    const previousState = state.trainingState;
    state.trainingState = training.state;


    // For idle state, only re-render if state changed or selectedGestureId changed
    if (training.state === 'idle') {
        const idleHash = `idle:${state.selectedGestureId}`;
        if (idleHash === state.lastTrainingHash) {
            return; // No change, skip re-render
        }
        state.lastTrainingHash = idleHash;
    } else {
        // For active states, always update (to show timers)
        state.lastTrainingHash = null;
    }

    // Remove all state classes
    elements.trainingPanel.classList.remove('training-countdown', 'training-capturing', 'training-resting');

    switch (training.state) {
        case 'idle':
            elements.qualityFeedback.classList.add('hidden');
            elements.trainingDisplay.innerHTML = `
                <button id="btn-start-training" class="big-button" ${!state.selectedGestureId ? 'disabled' : ''}>
                    ▶ START TRAINING
                </button>
                <p id="training-hint">${state.selectedGestureId ? 'Press [Space] or click to train' : 'Select a gesture above to train'}</p>
            `;
            // Event listener is handled via delegation in setupEventListeners
            elements.trainingStatus.textContent = 'IDLE';
            elements.trainingStatus.className = 'dim';
            break;

        case 'countdown':
            elements.trainingPanel.classList.add('training-countdown');
            elements.trainingDisplay.innerHTML = `
                <span class="countdown-number">${training.countdown}</span>
                <p>Get ready for rep ${training.current_rep} of ${training.total_reps}</p>
                <button class="stop-training-btn">Stop (Esc)</button>
            `;
            elements.trainingStatus.textContent = 'COUNTDOWN';
            elements.trainingStatus.className = 'gold';
            break;

        case 'capturing':
            elements.trainingPanel.classList.add('training-capturing');
            elements.trainingDisplay.innerHTML = `
                <span class="capture-indicator">███ CAPTURING ███</span>
                <span style="font-size: 48px; font-weight: 700; color: var(--red);">${Math.ceil(training.remaining)}s</span>
                <progress value="${training.progress}" max="1" style="width: 80%; height: 8px;"></progress>
                <p>${training.frame_count} frames captured</p>
                <button class="stop-training-btn">Stop (Esc)</button>
            `;
            elements.trainingStatus.textContent = 'CAPTURING';
            elements.trainingStatus.className = 'red';
            break;

        case 'resting':
            elements.trainingPanel.classList.add('training-resting');
            elements.trainingDisplay.innerHTML = `
                <span class="rest-indicator">REST</span>
                <span style="font-size: 32px;">${Math.ceil(training.remaining)}s</span>
                <p class="green">Completed ${training.completed_reps} of ${training.total_reps} reps</p>
                <button class="stop-training-btn">Stop (Esc)</button>
            `;
            elements.trainingStatus.textContent = 'RESTING';
            elements.trainingStatus.className = 'orange';
            break;

        case 'complete':
            // Show calibration info after training completes
            const trainedGesture = training.gesture_id && state.vocabulary?.gestures
                ? state.vocabulary.gestures.find(g => g.id === training.gesture_id)
                : null;
            const calibrationInfo = trainedGesture
                ? `Threshold auto-set to ${Math.round(trainedGesture.threshold)}`
                : 'Calibration complete';

            elements.trainingDisplay.innerHTML = `
                <span style="font-size: 28px; font-weight: 700; color: var(--green);">✓ COMPLETE!</span>
                <p class="dim">${calibrationInfo}</p>
            `;
            elements.trainingStatus.textContent = 'COMPLETE';
            elements.trainingStatus.className = 'green';

            // Show quality feedback if any issues detected
            if (training.quality_issues && training.quality_issues.length > 0) {
                elements.qualityFeedback.classList.remove('hidden');
                elements.qualityFeedback.innerHTML = training.quality_issues
                    .map(q => `<div class="quality-warning"><span class="quality-label">${q.label}</span> Example #${q.example_index}: ${q.message}</div>`)
                    .join('');
            } else {
                elements.qualityFeedback.classList.add('hidden');
                elements.qualityFeedback.innerHTML = '';
            }
            break;
    }
}

function updateMonitor(monitor) {
    if (!monitor) return;

    elements.recognizerStatus.textContent = monitor.active ? 'ACTIVE' : 'STOPPED';
    elements.recognizerStatus.className = monitor.active ? 'green' : 'red';
    elements.bufferStatus.textContent = `Buffer: ${monitor.buffer_len}`;
    elements.windowStatus.textContent = `Win: ${monitor.window_size}`;
    elements.exampleStatus.textContent = `Ex: ${monitor.total_examples}`;

    // Render gesture monitors
    elements.monitorList.innerHTML = '';
    for (const g of monitor.gestures) {
        const row = document.createElement('div');
        row.className = 'monitor-row';

        const distanceColor = g.distance !== null && g.distance < g.threshold ? 'green' : '';
        const distanceText = g.distance !== null ? Math.round(g.distance) : '...';

        // Dynamic slider range based on threshold (for MAN mode)
        const sliderMin = Math.max(100, Math.floor(g.threshold * 0.1));
        const sliderMax = Math.max(1000, Math.ceil(g.threshold * 3));
        const sliderStep = Math.max(1, Math.floor((sliderMax - sliderMin) / 100));

        // Show stats when in AUTO mode
        const statsDisplay = g.auto_mode && g.distance_mean != null
            ? `<span class="stats-hint dim">μ=${Math.round(g.distance_mean)} σ=${Math.round(g.distance_std)}</span>`
            : '';

        // AUTO mode: clean display with Tune button
        // MAN mode: slider visible for manual adjustment
        const thresholdControl = g.auto_mode
            ? `<span class="threshold-value">${Math.round(g.threshold)}</span>
               ${statsDisplay}
               <button class="btn-tune small" data-id="${g.id}">Tune</button>`
            : `<input type="range" min="${sliderMin}" max="${sliderMax}" step="${sliderStep}"
                      value="${Math.round(g.threshold)}" data-id="${g.id}" class="threshold-slider">
               <span class="threshold-value">${Math.round(g.threshold)}</span>
               <button class="btn-auto small" data-id="${g.id}">Auto</button>`;

        const consensusBtn = g.consensus_enabled
            ? '<button class="btn-consensus small active" data-id="' + g.id + '">CON</button>'
            : '<button class="btn-consensus small" data-id="' + g.id + '">CON</button>';

        // Distance bar: shows how close distance is to threshold (1.0 = at threshold)
        const ratio = (g.distance !== null && g.threshold > 0)
            ? Math.min(1.0, g.distance / g.threshold)
            : 1.0;
        const barPercent = Math.round((1.0 - ratio) * 100); // invert: close = full
        const barColor = ratio < 0.5 ? 'var(--green)' : ratio < 0.8 ? 'var(--yellow, #c6a832)' : 'var(--dim)';

        row.innerHTML = `
            <span class="gesture-name col-gesture">${g.name} (${g.example_count})</span>
            <span class="distance col-distance ${distanceColor}">${distanceText}</span>
            <span class="threshold-control col-threshold">${thresholdControl}</span>
            <span class="hit-indicator col-hit ${g.recent_hit ? 'green' : ''}">
                ${g.recent_hit ? '● HIT' : ''}
            </span>
            <span class="col-consensus">${consensusBtn}</span>
            <div class="distance-bar"><div class="distance-bar-fill" style="width:${barPercent}%;background:${barColor}"></div></div>
        `;

        // Event listeners based on mode
        if (g.auto_mode) {
            row.querySelector('.btn-tune').addEventListener('click', async () => {
                await invoke('toggle_threshold_mode', { gestureId: g.id });
            });
        } else {
            row.querySelector('.threshold-slider').addEventListener('input', async (e) => {
                const value = parseFloat(e.target.value);
                row.querySelector('.threshold-value').textContent = Math.round(value);
                await invoke('set_threshold', { gestureId: g.id, threshold: value, manual: true });
            });
            row.querySelector('.btn-auto').addEventListener('click', async () => {
                await invoke('toggle_threshold_mode', { gestureId: g.id });
            });
        }

        // Consensus toggle
        row.querySelector('.btn-consensus').addEventListener('click', async () => {
            await invoke('set_consensus', { gestureId: g.id, enabled: !g.consensus_enabled });
        });

        elements.monitorList.appendChild(row);
    }

    // Update hit display
    if (monitor.recent_hit) {
        elements.hitDisplay.textContent = `● ${monitor.recent_hit}`;
        elements.hitDisplay.className = 'active';
    } else {
        elements.hitDisplay.textContent = '—';
        elements.hitDisplay.className = '';
    }
}

function updateHitLog(hitLog) {
    if (!hitLog) return;

    elements.hitTotal.textContent = `${hitLog.total} total`;

    if (hitLog.entries.length === 0) {
        elements.hitLog.innerHTML = `
            <p class="dim">No hits yet</p>
            <p class="dim small">Perform a trained gesture</p>
        `;
        return;
    }

    elements.hitLog.innerHTML = '';
    for (const entry of hitLog.entries) {
        const div = document.createElement('div');
        div.className = `hit-entry ${entry.recent ? 'recent' : ''}`;

        const timeText = entry.ms_ago < 1000 ? 'NOW' : `${Math.floor(entry.ms_ago / 1000)}s`;

        div.innerHTML = `
            <span class="time">${timeText}</span>
            <span class="name">${entry.name}</span>
        `;
        elements.hitLog.appendChild(div);
    }
}

// Actions
function setMode(mode) {
    state.mode = mode;

    elements.btnTraining.classList.toggle('active', mode === 'training');
    elements.btnPerformance.classList.toggle('active', mode === 'performance');

    elements.trainingPanel.classList.toggle('hidden', mode !== 'training');
    elements.performancePanel.classList.toggle('hidden', mode !== 'performance');

    invoke('set_mode', { mode });
}

async function newVocabulary() {
    await invoke('new_vocabulary');
}

async function openVocabulary() {
    await invoke('open_vocabulary');
}

async function saveVocabulary() {
    await invoke('save_vocabulary');
}

async function sendTestHit() {
    await invoke('send_test_hit');
}

async function addGesture() {
    const id = await invoke('add_gesture');
    state.selectedGestureId = id;
    state.lastGesturesHash = null; // Force re-render
    state.lastTrainingHash = null; // Force training panel update
}

async function selectGesture(id) {
    state.selectedGestureId = id;
    state.lastGesturesHash = null; // Force re-render to show selection
    state.lastTrainingHash = null; // Force training panel update (button enable/disable)
    await invoke('select_gesture', { gestureId: id });
}

async function renameGesture(id, nameElement) {
    const gesture = state.vocabulary.gestures.find(g => g.id === id);
    if (!gesture) return;

    // Prevent re-renders while editing
    state.isEditingGestureName = true;

    // Create an input field for inline editing
    const input = document.createElement('input');
    input.type = 'text';
    input.value = gesture.name;
    input.className = 'inline-edit';

    // Replace the name element with the input
    const originalText = nameElement.textContent;
    nameElement.textContent = '';
    nameElement.appendChild(input);
    input.focus();
    input.select();

    // Clean up function
    const finishEditing = () => {
        state.isEditingGestureName = false;
        state.lastGesturesHash = null; // Force re-render on next poll
    };

    // Handle saving
    const save = async () => {
        const newName = input.value.trim();
        if (newName && newName !== gesture.name) {
            await invoke('rename_gesture', { gestureId: id, name: newName });
        }
        finishEditing();
    };

    // Save on blur or Enter
    input.addEventListener('blur', save);
    input.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            input.blur();
        } else if (e.key === 'Escape') {
            nameElement.textContent = originalText;
            finishEditing();
        }
    });
}

async function deleteGesture(id) {
    const gesture = state.vocabulary.gestures.find(g => g.id === id);
    if (!gesture) return;

    if (confirm(`Delete "${gesture.name}"?`)) {
        try {
            await invoke('delete_gesture', { gestureId: id });
            if (state.selectedGestureId === id) {
                state.selectedGestureId = null;
            }
            state.lastGesturesHash = null; // Force re-render
        } catch (e) {
            console.error('Failed to delete gesture:', e);
        }
    }
}

function toggleExampleList(gestureId) {
    if (state.expandedGestures.has(gestureId)) {
        state.expandedGestures.delete(gestureId);
    } else {
        state.expandedGestures.add(gestureId);
    }
    state.lastGesturesHash = null; // Force re-render
}

async function deleteExample(gestureId, exampleIndex) {
    try {
        await invoke('delete_example', { gestureId, exampleIndex });
        state.lastGesturesHash = null;
    } catch (err) {
        console.error('delete_example failed:', err);
    }
}

async function startTraining() {
    if (!state.selectedGestureId) return;
    if (state.trainingState !== 'idle') return;

    await invoke('start_training', {
        gestureId: state.selectedGestureId,
        reps: parseInt(elements.trainReps.value),
        countdownSecs: parseInt(elements.trainCountdown.value),
        durationSecs: parseInt(elements.trainDuration.value),
        restSecs: parseInt(elements.trainRest.value),
    });
}

async function cancelTraining() {
    if (state.trainingState !== 'idle') {
        await invoke('cancel_training');
    }
}

async function toggleAugmentation() {
    // Read current state from the button text (ON/OFF)
    const isCurrentlyEnabled = elements.btnAugmentation.textContent !== 'OFF';
    await invoke('set_augmentation_enabled', { enabled: !isCurrentlyEnabled });
}

async function toggleJointWeighting() {
    const isCurrentlyEnabled = elements.btnJointWeighting.textContent !== 'OFF';
    await invoke('set_joint_weighting', { enabled: !isCurrentlyEnabled });
}

async function toggleComplexityCorrection() {
    const isCurrentlyEnabled = elements.btnComplexityCorrection.textContent !== 'OFF';
    await invoke('set_complexity_correction', { enabled: !isCurrentlyEnabled });
}

async function toggleF1Threshold() {
    const isCurrentlyEnabled = elements.btnF1Threshold.textContent !== 'OFF';
    await invoke('set_f1_threshold', { enabled: !isCurrentlyEnabled });
}

async function toggleDiagnostics() {
    if (state.diagnosticsEnabled) {
        // Stop recording
        await invoke('disable_diagnostics');
        state.diagnosticsEnabled = false;
        elements.btnToggleDiagnostics.textContent = 'Start Recording';
        elements.btnToggleDiagnostics.classList.remove('recording');
        elements.diagnosticsStatus.textContent = `Saved: ${state.diagnosticsPath}`;
        elements.diagnosticsStatus.className = 'green';
    } else {
        // Start recording - backend generates the path
        try {
            const path = await invoke('enable_diagnostics');
            state.diagnosticsEnabled = true;
            state.diagnosticsPath = path;
            elements.btnToggleDiagnostics.textContent = 'Stop Recording';
            elements.btnToggleDiagnostics.classList.add('recording');
            elements.diagnosticsStatus.textContent = 'Recording...';
            elements.diagnosticsStatus.className = 'red';
        } catch (e) {
            elements.diagnosticsStatus.textContent = `Error: ${e}`;
            elements.diagnosticsStatus.className = 'red';
        }
    }
}
