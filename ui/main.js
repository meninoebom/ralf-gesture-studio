// RALF Gesture Studio - Frontend JavaScript
console.log('main.js loading...');

const { invoke } = window.__TAURI__.core;
console.log('Tauri invoke loaded:', typeof invoke);

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
};

// DOM Elements
const elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded fired');
    cacheElements();
    console.log('Elements cached');
    setupEventListeners();
    console.log('Event listeners set up');
    await loadInitialState();
    console.log('Initial state loaded');
    startPolling();
    console.log('Polling started');
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

    elements.inputStatus = document.getElementById('input-status');
    elements.inputStatusDot = document.getElementById('input-status-dot');
    elements.inputDetail = document.getElementById('input-detail');
    elements.frameCount = document.getElementById('frame-count');
    elements.outputStatus = document.getElementById('output-status');
    elements.outputStatusDot = document.getElementById('output-status-dot');
    elements.outputDetail = document.getElementById('output-detail');
    elements.sendCount = document.getElementById('send-count');
    elements.btnTestHit = document.getElementById('btn-test-hit');

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
    elements.btnTestHit.addEventListener('click', sendTestHit);

    // Gestures
    elements.btnAddGesture.addEventListener('click', addGesture);

    // Training button is dynamically rendered, use event delegation
    elements.trainingDisplay.addEventListener('click', (e) => {
        console.log('Training display clicked, target:', e.target.id);
        if (e.target.id === 'btn-start-training') {
            console.log('Start training button detected via delegation');
            startTraining();
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        console.log('Keydown:', e.code, 'mode:', state.mode, 'trainingState:', state.trainingState);
        if (e.code === 'Space' && state.mode === 'training' && state.trainingState === 'idle') {
            e.preventDefault();
            console.log('Space pressed, calling startTraining');
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

    // Training state
    updateTrainingState(appState.training);

    // Performance mode data
    if (state.mode === 'performance') {
        updateMonitor(appState.monitor);
        updateHitLog(appState.hit_log);
    }
}

function updateConnectionStatus(status) {
    if (!status) return;

    // Input
    const inputColors = {
        'Stopped': ['gray', 'STOPPED', ''],
        'Listening': ['gold', 'LISTENING', '(waiting for data)'],
        'Receiving': ['green', 'RECEIVING', `(${(status.ms_since_last_frame / 1000).toFixed(1)}s ago)`],
        'Error': ['red', 'ERROR', `(${status.input_error || 'Unknown error'})`],
    };

    const [dotClass, statusText, detail] = inputColors[status.input_status] || ['gray', 'UNKNOWN', ''];
    elements.inputStatusDot.className = `status-dot ${dotClass}`;
    elements.inputStatus.textContent = statusText;
    elements.inputStatus.className = dotClass;
    elements.inputDetail.textContent = detail;
    elements.frameCount.textContent = `Frames: ${status.frame_count}`;

    // Output
    const outputColors = {
        'Ready': ['green', 'READY', ''],
        'Sent': ['green', 'SENT', `(${(status.ms_since_last_send / 1000).toFixed(1)}s ago)`],
        'Error': ['red', 'ERROR', `(${status.output_error || 'Unknown error'})`],
    };

    const [outDot, outStatus, outDetail] = outputColors[status.output_status] || ['green', 'READY', ''];
    elements.outputStatusDot.className = `status-dot ${outDot}`;
    elements.outputStatus.textContent = outStatus;
    elements.outputStatus.className = outDot;
    elements.outputDetail.textContent = outDetail;
    elements.sendCount.textContent = `Sent: ${status.send_count}`;
}

function renderGestures(gestures) {
    // Skip re-render if currently editing a gesture name
    if (state.isEditingGestureName) {
        // Still update the training panel's selected gesture display
        const selected = gestures.find(g => g.id === state.selectedGestureId);
        elements.selectedGesture.textContent = selected ? selected.name : '(none)';
        return;
    }

    // Create a hash of the gesture data + selected ID to detect changes
    const currentHash = JSON.stringify({
        gestures: gestures.map(g => ({ id: g.id, name: g.name, examples: g.examples.length, osc_address: g.osc_address })),
        selectedId: state.selectedGestureId
    });

    // Skip re-render if nothing changed
    if (currentHash === state.lastGesturesHash) {
        return;
    }
    state.lastGesturesHash = currentHash;

    elements.gestureList.innerHTML = '';

    for (const gesture of gestures) {
        const row = document.createElement('div');
        row.className = 'gesture-row';
        row.innerHTML = `
            <span class="col-status">
                <span class="status-dot ${gesture.examples.length > 0 ? 'green' : 'gray'}">
                    ${gesture.examples.length > 0 ? '●' : '○'}
                </span>
            </span>
            <span class="name col-name ${state.selectedGestureId === gesture.id ? 'selected' : ''}"
                  data-id="${gesture.id}">${gesture.name}</span>
            <span class="examples col-examples">${gesture.examples.length}</span>
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
        row.querySelector('.btn-select').addEventListener('click', (e) => {
            e.stopPropagation();
            selectGesture(gesture.id);
        });
        row.querySelector('.btn-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteGesture(gesture.id);
        });

        elements.gestureList.appendChild(row);
    }

    // Update selected gesture name in training panel
    const selected = gestures.find(g => g.id === state.selectedGestureId);
    elements.selectedGesture.textContent = selected ? selected.name : '(none)';
}

function updateTrainingState(training) {
    if (!training) return;

    const previousState = state.trainingState;
    state.trainingState = training.state;

    // Log state changes
    if (previousState !== training.state) {
        console.log('Training state changed:', previousState, '->', training.state);
    }

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
                <p class="dim">Press [Esc] to cancel</p>
            `;
            elements.trainingStatus.textContent = 'COUNTDOWN';
            elements.trainingStatus.className = 'gold';
            break;

        case 'capturing':
            elements.trainingPanel.classList.add('training-capturing');
            elements.trainingDisplay.innerHTML = `
                <span class="capture-indicator">███ CAPTURING ███</span>
                <span style="font-size: 48px; font-weight: 700; color: var(--red);">${training.remaining.toFixed(1)}s</span>
                <progress value="${training.progress}" max="1" style="width: 80%; height: 8px;"></progress>
                <p>${training.frame_count} frames captured</p>
                <p class="dim">Press [Esc] to cancel</p>
            `;
            elements.trainingStatus.textContent = 'CAPTURING';
            elements.trainingStatus.className = 'red';
            break;

        case 'resting':
            elements.trainingPanel.classList.add('training-resting');
            elements.trainingDisplay.innerHTML = `
                <span class="rest-indicator">REST</span>
                <span style="font-size: 32px;">${training.remaining.toFixed(1)}s</span>
                <p class="green">Completed ${training.completed_reps} of ${training.total_reps} reps</p>
                <p class="dim">Press [Esc] to cancel</p>
            `;
            elements.trainingStatus.textContent = 'RESTING';
            elements.trainingStatus.className = 'orange';
            break;

        case 'complete':
            elements.trainingDisplay.innerHTML = `
                <span style="font-size: 28px; font-weight: 700; color: var(--green);">COMPLETE!</span>
            `;
            elements.trainingStatus.textContent = 'COMPLETE';
            elements.trainingStatus.className = 'green';
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

        row.innerHTML = `
            <span class="gesture-name col-gesture">${g.name} (${g.example_count})</span>
            <span class="distance col-distance ${distanceColor}">${distanceText}</span>
            <span class="threshold-control col-threshold">
                <input type="range" min="10" max="500" value="${g.threshold}"
                       data-id="${g.id}" class="threshold-slider">
                <span class="threshold-value">${Math.round(g.threshold)}</span>
            </span>
            <span class="mode-toggle col-mode ${g.auto_mode ? 'blue' : 'dim'}" data-id="${g.id}">
                ${g.auto_mode ? 'AUTO' : 'MAN'}
            </span>
            <span class="hit-indicator col-hit ${g.recent_hit ? 'green' : ''}">
                ${g.recent_hit ? '● HIT' : ''}
            </span>
        `;

        row.querySelector('.threshold-slider').addEventListener('input', async (e) => {
            const value = parseFloat(e.target.value);
            row.querySelector('.threshold-value').textContent = Math.round(value);
            await invoke('set_threshold', { gestureId: g.id, threshold: value, manual: true });
        });

        row.querySelector('.mode-toggle').addEventListener('click', async () => {
            await invoke('toggle_threshold_mode', { gestureId: g.id });
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
        await invoke('delete_gesture', { gestureId: id });
        if (state.selectedGestureId === id) {
            state.selectedGestureId = null;
        }
        state.lastGesturesHash = null; // Force re-render
    }
}

async function startTraining() {
    console.log('startTraining called', {
        selectedGestureId: state.selectedGestureId,
        trainingState: state.trainingState
    });

    if (!state.selectedGestureId) {
        console.log('No gesture selected, returning');
        return;
    }
    if (state.trainingState !== 'idle') {
        console.log('Training state is not idle:', state.trainingState);
        return;
    }

    const params = {
        gestureId: state.selectedGestureId,
        reps: parseInt(elements.trainReps.value),
        countdownSecs: parseInt(elements.trainCountdown.value),
        durationSecs: parseInt(elements.trainDuration.value),
        restSecs: parseInt(elements.trainRest.value),
    };
    console.log('Invoking start_training with:', params);

    try {
        await invoke('start_training', params);
        console.log('start_training invoke succeeded');
    } catch (e) {
        console.error('start_training invoke failed:', e);
    }
}

async function cancelTraining() {
    if (state.trainingState !== 'idle') {
        await invoke('cancel_training');
    }
}
