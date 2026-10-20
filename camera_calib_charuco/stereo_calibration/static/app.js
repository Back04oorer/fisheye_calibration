import {SessionPicker} from './sessions.js';
import {PointInspector} from './inspection.js';
import {VideoPreview} from './video.js';
import {updateBoardGraphic, updateMovementGuide} from './movement_guide.js';
const inspector = new PointInspector();
const $ = (id) => document.getElementById(id);
const show = (id, visible) => {
    $(id).hidden = !visible;
};
const set = (id, text) => {
    if ($(id).textContent !== String(text)) $(id).textContent = text;
};
const number = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
let state, presets, initialized = false,
    dirty = false,
    busy = false;
let toastTimer;
let savedViewsKey, goalsKey, resultsKey;
let foldersKey;
const sessions = new SessionPicker({request, getState: () => state, onSwitch: update});
const video = new VideoPreview({getState: () => state,
    onFrame: (s, mode, matched) => inspector.update(s, mode, matched),
    onError: error => set('frameRate', error.message)});
const movementTitle = (title) => title.replace(/^Check /, '').replace(/^./, char => char.toUpperCase());


function toast(message) {
    set('toast', message);
    show('toast', true);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => show('toast', false), 7000);
}

async function request(path, body) {
    const response = await fetch(path, body ? {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({...body, workspace_id: state?.workspace_id}),
    } : {});
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status})`);
    }
    return response.json();
}

async function updateFolders(s) {
    const key = JSON.stringify([s.workspace_id, s.session, s.inspection_recording?.path]);
    if (key === foldersKey) return;
    foldersKey = key;
    const buttons = [...document.querySelectorAll('[data-folder]')];
    buttons.forEach(button => button.hidden = true);
    try {
        const folders = await request('/api/folders');
        if (foldersKey !== key) return;
        buttons.forEach(button => {
            const path = folders[button.dataset.folder];
            button.hidden = !path;
            button.title = path || '';
        });
    } catch (error) {
        if (foldersKey === key) foldersKey = null;
    }
}

document.querySelectorAll('[data-folder]').forEach(button => {
    button.addEventListener('click', async () => {
        button.disabled = true;
        try {
            await request('/api/folders/open', {kind: button.dataset.folder});
        } catch (error) {
            toast(error.message);
        } finally {
            button.disabled = false;
        }
    });
});

function selectedBoard() {
    const preset = presets[$('preset').value];
    const square = Number($('square').value);
    return {
        ...preset,
        square_mm: square,
        marker_mm: square * preset.marker_mm / preset.square_mm
    };
}

function boardInfo() {
    if (!presets) return;
    const b = selectedBoard();
    set('boardMeta', `${b.columns} × ${b.rows} squares · ${number(b.square_mm)} / ${number(b.marker_mm)} mm · ${b.dictionary} · legacy layout`);
    $('boardPreview').src = `/api/board.png?preset=${encodeURIComponent($('preset').value)}`;
    $('downloadBoard').href = `/api/board.pdf?preset=${encodeURIComponent($('preset').value)}`;
}

function initializeForm(s) {
    const preset = Object.keys(presets).find(key => presets[key].columns === s.board.columns && presets[key].rows === s.board.rows);
    $('preset').value = preset || 'existing-20mm';
    $('square').value = s.board.square_mm;
    for (const key of ['device', 'width', 'height', 'fps']) $(key).value = s.camera[key];
    $('swap').checked = s.camera.swap;
    boardInfo();
    initialized = true;
}

function heatmap(id, values) {
    const grid = $(id);
    const key = JSON.stringify(values);
    if (grid.dataset.values === key) return;
    grid.dataset.values = key;
    grid.replaceChildren();
    const maximum = Math.max(1, ...values.flat());
    for (const count of values.flat()) {
        const cell = document.createElement('span');
        cell.textContent = count || '·';
        if (count) cell.style.background = `rgba(103,153,100,${.13 + .50 * Math.sqrt(count / maximum)})`;
        cell.title = `${count} detected training corners`;
        grid.append(cell);
    }
}

function updateGoals(s) {
    const goals = s.report?.regions || s.goals;
    const key = JSON.stringify([goals, s.goal?.key, s.training_count, s.validation_count]);
    if (key === goalsKey) return;
    goalsKey = key;
    const isDone = goal => goal.status ? goal.status === 'passed' : goal.done;
    set('goalsLabel', goals.some(goal => goal.key.startsWith('check-')) ? 'Accuracy checks' : 'Movements');
    $('goals').replaceChildren();
    for (const goal of goals) {
        const chip = document.createElement('span');
        chip.textContent = `${isDone(goal) ? '✓ ' : goal.status === 'failed' ? '! ' : ''}${movementTitle(goal.title)}`;
        chip.classList.toggle('done', isDone(goal));
        chip.classList.toggle('failed', goal.status === 'failed');
        chip.classList.toggle('current', goal.key.replace(/^check-/, '') === s.goal?.key.replace(/^check-/, ''));
        chip.title = ({passed:'Passed under this fit', failed:'Saved measurement still fails', confirm:'Confirm the corrected area with a new image', missing:'No check captured here yet'})[goal.status] || '';
        $('goals').append(chip);
    }
    set('goalCount', `${goals.filter(g => !isDone(g)).length} remaining`);
    set('trainingCount', s.training_count);
    set('validationCount', s.validation_count);
}

function updateSavedViews(s) {
    const views = s.saved_views || [];
    show('savedPanel', views.length > 0);
    const key = JSON.stringify([s.session, views]);
    if (key === savedViewsKey) return;
    savedViewsKey = key;
    set('savedCount', `${views.length} saved pairs`);
    const used = views.filter(v => v.status === 'used').length;
    const checks = views.filter(v => v.role === 'validation').length;
    set('savedUsage', `${used ? `${used} used in fit` : `${s.training_count} for fitting`} · ${checks} check${checks === 1 ? '' : 's'}`);
    const labels = {used: 'In fit', training: 'For fitting', passed: 'Check passed', failed: 'Check failed', pending: 'Check saved', excluded: 'Undone'};
    $('savedViews').replaceChildren();
    views.forEach((view, index) => {
        const link = document.createElement('a');
        link.href = `/api/captures/${encodeURIComponent(view.id)}?thumbnail=false`;
        link.title = `Download original stereo pair · ${view.id}`;
        const image = document.createElement('img');
        image.src = `/api/captures/${encodeURIComponent(view.id)}`;
        image.alt = `Saved stereo pair ${index + 1}`;
        image.loading = 'lazy';
        const label = document.createElement('span');
        label.textContent = `${index + 1} · ${labels[view.status]}`;
        link.append(image, label);
        $('savedViews').append(link);
    });
}

function updateGuide(s) {
    const labels = {
        ready: 'Board check',
        collecting: 'Collecting',
        fitting: 'Fitting',
        validating: 'Accuracy check',
        done: 'Complete',
        needs_attention: 'Review needed',
        replay: 'Replay'
    };
    const refining = s.phase === 'collecting' && !!s.refinement_reason;
    set('phase', s.paused && s.phase !== 'done' ? 'Paused' : refining ? 'Refining' : labels[s.phase] || s.phase);
    const step = ({
        ready: 'setup',
        collecting: 'collect',
        fitting: 'collect',
        validating: 'collect',
        done: 'result',
        needs_attention: 'result',
        replay: 'result'
    })[s.phase];
    document.querySelectorAll('[data-step]').forEach(el => el.classList.toggle('active', el.dataset.step === step));
    const adjustment = s.adjustment || {kind: 'unavailable'};
    const collecting = !s.replay && ['collecting', 'validating'].includes(s.phase);
    const checking = !s.replay && s.phase === 'validating';
    const holding = collecting && !s.paused && adjustment.kind === 'ready';
    let title = s.goal ? movementTitle(s.goal.title) : 'Vary the board pose';
    if (checking && adjustment.kind === 'move' && s.goal) title = `Verify ${movementTitle(s.goal.title).toLowerCase()}`;
    if (refining && adjustment.kind === 'move' && s.goal) title = `Improve ${movementTitle(s.goal.title).toLowerCase()}`;
    let instruction = 'Keep the board visible in both cameras.';
    if (adjustment.kind === 'move') {
        instruction = 'Move the board centre into the blue area.';
    } else if (adjustment.kind === 'tilt') {
        title = `Tilt ${adjustment.edge} edge`;
        instruction = `Bring the printed ${adjustment.edge} edge toward the cameras.`;
    } else if (adjustment.kind === 'distance') {
        title = adjustment.direction === 'closer' ? 'Come closer' : 'Move farther';
        instruction = 'Keep the board in this area. Change its distance.';
    } else if (holding) {
        title = 'Hold still';
        instruction = 'Capturing automatically when steady.';
    }
    if (collecting && s.common_corners < 12) {
        instruction = 'Bring the board closer so both cameras can detect it.';
    }
    if (s.phase === 'ready') {
        title = 'Show your board';
        instruction = 'Show your board to both cameras.';
    }
    if (s.phase === 'fitting') {
        title = 'Calculating';
        instruction = 'Pose saved — you can move the board. Keep the cameras fixed while the fit is calculated.';
    }
    if (s.phase === 'done') {
        title = 'Geometric checks passed';
        instruction = 'Calibration ready to download.';
    }
    if (s.phase === 'needs_attention') {
        title = 'Some checks need attention';
        instruction = 'Check board flatness and lighting.';
    }
    if (s.replay) {
        title = 'Review a saved session';
        instruction = 'Read-only replay.';
    }
    set('goalTitle', title);
    const reused = s.report?.reused_passed_regions?.length || 0;
    show('guideContext', checking || refining);
    set('guideContext', refining ? 'A new pose here will help refine the fit.' : reused ? 'Saved checks reused. Only remaining areas need capture.' : checking ? 'New image to verify the existing fit.' : '');
    set('instruction', instruction);
    show('illustration', collecting && !!s.goal);
    set('guideLabel', s.phase === 'done' ? 'RESULT' : holding ? 'AUTO CAPTURE' : 'NEXT MOVEMENT');
    updateMovementGuide(adjustment, s.board, {holding, paused:s.paused});
    let captureMessage = s.message;
    if (s.paused) captureMessage = 'Paused';
    else if (s.phase === 'fitting') captureMessage = 'Calculating — you can move the board';
    else if (s.just_captured && !holding) captureMessage = 'Captured — move to the next pose';
    else if (s.common_corners < 12) captureMessage = 'Bring the board closer.';
    else if (holding) captureMessage = 'Hold still';
    else if (s.message.startsWith('This pose is already covered')) captureMessage = 'Follow the next movement.';
    else if (s.message === 'Board detected in both cameras.') captureMessage = 'Board detected';
    set('captureMessage', captureMessage);
    show('captureStatus', !s.replay && s.phase !== 'done');
    $('captureMessage').title = s.message;
    set('holdPercent', s.hold_progress > 0 ? `${Math.round(s.hold_progress * 100)}%` : '');
    $('holdFill').style.width = `${s.hold_progress * 100}%`;
    show('start', !s.configured && !s.replay);
    $('start').disabled = dirty || busy || s.common_corners < 12 || !!s.error;
    show('pause', s.configured && !['done', 'needs_attention'].includes(s.phase));
    $('pause').dataset.action = s.paused ? 'resume' : 'pause';
    set('pause', s.paused ? 'Resume' : 'Pause');
    show('undo', s.phase === 'collecting' && s.training_count > 0 && !s.replay);
    show('more', ['done', 'needs_attention'].includes(s.phase) && !s.replay);
    show('reconnect', !!s.error && !s.replay);
    $('fit').disabled = s.phase !== 'collecting' || !s.can_fit || busy || s.replay;
}

function updateResults(s) {
    const key = JSON.stringify([s.has_model, s.phase, s.report, s.candidate, s.fit_rms, s.refinement_reason]);
    if (key === resultsKey) return;
    resultsKey = key;
    show('results', s.has_model || !!s.report);
    set('fitRms', s.fit_rms ? `${s.fit_rms.map(v => number(v, 3)).join(' / ')} px` : '—');
    const report = s.report;
    const passed = report?.passed && !s.refinement_reason;
    const failures = report?.validation_views?.filter(v => !v.passed).length || 0;
    const complete = report?.goals_complete;
    const fitFailed = report && !report.fit_passed;
    const checking = s.phase === 'validating' && !passed && !fitFailed && !failures;
    set('resultTitle', s.refinement_reason ? 'Refining calibration' : passed ? 'Geometric checks passed' : fitFailed || failures ? 'Retrying' : checking ? 'Checking accuracy' : 'Keep following the guide');
    set('resultBadge', passed ? 'Passed' : complete || failures || fitFailed ? 'Needs attention' : 'Pending');
    $('resultBadge').classList.toggle('warning', !!failures || !!fitFailed);
    const regions = report?.regions || [];
    const summary = [['passed', 'passed'], ['failed', 'to improve'], ['confirm', 'to confirm'], ['missing', 'unchecked']]
        .map(([status, label]) => {
            const count = regions.filter(region => region.status === status).length;
            return count ? `${count} ${label}` : '';
        }).filter(Boolean).join(' · ');
    set('resultText', passed ? 'All saved checks pass under this calibration.' : regions.length ? `Areas: ${summary}` : checking ? 'Your saved captures built the fit. New images check each area.' : 'Capture, accuracy checks and retries run automatically.');
    set('retryReason', s.refinement_reason || '');
    show('retryReason', !!s.refinement_reason);
    set('baseline', report ? `${number(report.baseline_mm)} mm` : 'Pending');
    $('viewResults').replaceChildren();
    for (const [index, view] of(report?.validation_views || []).entries()) {
        const row = document.createElement('tr');
        const area = view.goals?.map(key => key.replace(/^check-/, '').replaceAll('-', ' ')).join(', ');
        for (const value of [`${index + 1} · ${area || view.name.slice(0, 8)}${view.reused ? ' · reused' : ''}`, `${number(view.left_reprojection_rms_px, 3)} / ${number(view.right_reprojection_rms_px, 3)}`, `${number(view.left_to_right_rms_px, 3)} / ${number(view.right_to_left_rms_px, 3)}`, view.passed ? 'Pass' : view.error || view.failed_checks?.join('; ') || 'Review']) {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.append(cell);
        }
        row.lastChild.className = view.passed ? 'pass' : 'fail';
        $('viewResults').append(row);
    }
    $('artifacts').replaceChildren();
    show('artifacts', !!passed);
    if (s.candidate)
        for (const [filename, label] of [
                ['stereo_params.npz', 'Calibration NPZ'],
                ['stereo_params.yaml', 'Calibration YAML'],
                ['report.json', 'Validation report']
            ]) {
            if (filename === 'report.json' && !report) continue;
            const link = document.createElement('a');
            link.href = `/api/artifacts/${s.candidate}/${filename}`;
            link.textContent = `${label} ↓`;
            $('artifacts').append(link);
        }
}

function update(s) {
    const changed = state?.workspace_id !== s.workspace_id;
    state = s;
    if (changed) {
        initialized = false;
        dirty = false;
        video.reset();
        savedViewsKey = null;
        goalsKey = null;
        resultsKey = null;
        guideBoardKey = null;
        inspector.clear('Waiting for frame');
        show('frame', false);
        show('emptyFrame', true);
        $('projection').value = 'raw';
        $('previewSource').value = s.replay ? 'measured' : 'live';
        $('previewSource').disabled = !!s.replay;
    }
    sessions.update(s);
    updateFolders(s);
    const picker = s.phase === 'picker';
    show('streamOnly', !picker);
    $('streamOnly').dataset.action = s.stream_only ? 'resume_detection' : 'stream_only';
    $('streamOnly').disabled = busy;
    set('streamOnly', s.stream_only ? 'Resume board detection' : 'Stream only');
    if (s.stream_only && $('previewSource').value !== 'live' && !s.replay) {
        $('previewSource').value = 'live';
        video.reset();
    }
    show('sessionLanding', picker);
    show('calibrationWorkspace', !picker);
    show('resultsShortcut', !!s.has_model);
    if (picker) {
        set('connection', 'Ready');
        $('connection').classList.add('online');
        return;
    }
    updateBoardGraphic(s.board);
    if (presets && !initialized) initializeForm(s);
    set('connection', s.stream_only ? 'Stream only' : s.replay ? 'Replay connected' : s.error ? 'Needs attention' : 'Connected');
    $('connection').classList.add('online');
    show('error', !!s.error);
    set('error', s.error || '');
    if (s.error || s.frame_id < 0) {
        inspector.clear(s.error ? 'Preview unavailable' : 'Waiting for frame');
        if (!$('frame').hidden) video.reset();
    }
    show('setupPanel', !s.configured && !s.replay);
    const recording = s.inspection_recording;
    show('inspectionRecording', !!recording);
    if (recording) {
        set('inspectionRecordingStatus', recording.error ? `Recording failed: ${recording.error}`
            : `${recording.active ? 'Recording' : 'Saved'} · ${recording.frames} ${recording.save_images ? 'images + readings' : 'detection frames'}`);
        $('inspectionRecordingStatus').title = recording.path;
        show('stopInspection', recording.active);
    }
    updateGuide(s);
    updateGoals(s);
    updateSavedViews(s);
    updateResults(s);
    for (const [i, eye] of ['left', 'right'].entries()) {
        set(`${eye}Corners`, s.quality[i]?.corners ?? '—');
        set(`${eye}Square`, `${number(s.quality[i]?.square_px, 1)} px / square`);
        heatmap(`${eye}Coverage`, s.coverage[eye]);
    }
    set('sharedCorners', s.common_corners);
    if (s.error) set('frameRate', 'Preview stopped · see error');
    if (!video.matched && !s.error) inspector.update(s, $('projection').value, false);
    $('frameWrap').style.setProperty('--stereo-aspect', `${s.camera.width}/${s.camera.height}`);
    $('frameWrap').style.setProperty('--eye-aspect', `${s.camera.width / 2}/${s.camera.height}`);
    set('sessionPath', s.session || 'A session is created when collection starts.');
    set('modeDetails', `${s.camera.width} × ${s.camera.height} stereo · ${s.camera.width/2} × ${s.camera.height} per eye · ${s.board.square_mm} mm squares · ${s.board.dictionary} · ${s.board.legacy ? 'legacy' : 'modern'} layout`);
    set('sharpness', s.quality.length ? `Recognized markers: left ${s.quality[0].markers ?? '—'}, right ${s.quality[1].markers ?? '—'}. Board-region sharpness (advisory): left ${number(s.quality[0].sharpness, 0)}, right ${number(s.quality[1].sharpness, 0)}. No universal sharpness cutoff is applied.` : '');
    $('projection').options[1].disabled = !s.has_model;
}

async function poll() {
    const start = performance.now();
    try {
        const revision = state?.workspace_id;
        const s = await request('/api/state');
        if (!sessions.busy && revision === state?.workspace_id) {
            update(s);
        }
    } catch (error) {
        inspector.clear('Preview disconnected');
        video.reset();
        set('connection', 'Disconnected');
        $('connection').classList.remove('online');
        show('error', true);
        set('error', `Connection lost. Captures remain on disk. ${error.message}`);
    }
    setTimeout(poll, Math.max(0, 100 - (performance.now() - start)));
}

document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    button.disabled = true;
    try {
        update(await request('/api/control', {
            action: button.dataset.action
        }));
    } catch (error) {
        toast(error.message);
    } finally {
        busy = false;
        button.disabled = false;
    }
}));

document.querySelectorAll('[data-eye]').forEach(button => {
    if (button.tagName !== 'BUTTON') return;
    button.addEventListener('click', () => {
        $('frameWrap').dataset.eye = button.dataset.eye;
        document.querySelectorAll('.eye-labels span').forEach((label, index) => {
            label.hidden = button.dataset.eye !== 'both' && button.dataset.eye !== ['left', 'right'][index];
            label.style.width = button.dataset.eye === 'both' ? '50%' : '100%';
        });
        document.querySelectorAll('button[data-eye]').forEach(el => el.setAttribute('aria-pressed', String(el === button)));
    });
});

$('setupForm').addEventListener('input', () => {
    dirty = true;
    set('settingsStatus', 'Apply your changes before starting');
    boardInfo();
});
$('preset').addEventListener('change', () => {
    $('square').value = presets[$('preset').value].square_mm;
    boardInfo();
});
$('setupForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    busy = true;
    let device = $('device').value.trim();
    if (/^\d+$/.test(device)) device = Number(device);
    const camera = {
        device,
        width: Number($('width').value),
        height: Number($('height').value),
        fps: Number($('fps').value),
        swap: $('swap').checked
    };
    try {
        update(await request('/api/configure', {
            board: selectedBoard(),
            camera
        }));
        dirty = false;
        set('settingsStatus', 'Settings applied');
        video.reset();
    } catch (error) {
        toast(error.message);
    } finally {
        busy = false;
    }
});

document.addEventListener('keydown', event => {
    if (!$('sessionDialog').open && event.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName) && !$('pause').hidden) {
        event.preventDefault();
        $('pause').click();
    }
});

try {
    const setup = await request('/api/setup');
    presets = setup.presets;
    for (const device of setup.devices) {
        const option = document.createElement('option');
        option.value = device.index;
        option.label = device.name;
        $('devices').append(option);
    }
} catch (error) {
    toast(error.message);
}
poll();
video.run();
