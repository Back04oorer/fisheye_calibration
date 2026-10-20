// Frame-matched read-only visualization; never sends capture or fitting commands.
const svgNS = 'http://www.w3.org/2000/svg';
const byId = id => document.getElementById(id);
const colour = {shared: '#70d746', left: '#51bdf1', right: '#ffbc48', missing: '#bdc5be'};
const element = (name, attributes) => {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
};
const px = value => Number.isFinite(value) ? `${value.toFixed(2)} px` : '—';

export class PointInspector {
    constructor({sourceId = 'previewSource', boardUrl = '/api/board.png', pointRadius = 5, overlayId = 'pointOverlay', selectMeasuredOnIds = true} = {}) {
        this.current = null;
        this.sourceId = sourceId;
        this.boardUrl = boardUrl;
        this.pointRadius = pointRadius;
        this.overlayId = overlayId;
        this.panel = byId('pointInspector');
        this.panel.addEventListener('toggle', () => this.draw());
        byId('showPointIds').addEventListener('change', () => {
            if (selectMeasuredOnIds && byId('showPointIds').checked) {
                byId(this.sourceId).value = 'measured';
                byId(this.sourceId).dispatchEvent(new Event('change'));
            }
            this.draw();
        });
    }

    setOverlay(id) {
        if (id === this.overlayId) return;
        byId(this.overlayId).replaceChildren();
        this.overlayId = id;
        this.draw();
    }

    update(state, mode, matched = true) {
        const data = state.inspection;
        if (!data || data.frame_id !== state.frame_id) {
            this.clear('Waiting for detection');
            return;
        }
        if (this.current?.data.frame_id === data.frame_id && this.current.mode === mode
            && this.current.matched === matched && this.current.state.workspace_id === state.workspace_id) return;
        this.current = {state, mode, data, matched};
        const left = data.points.left.length, right = data.points.right.length;
        const shared = data.shared_ids.length;
        for (const [id, count] of [['inspectLeft', left], ['inspectShared', shared], ['inspectRight', right]]) {
            byId(id).textContent = `${count} / ${data.total}`;
        }
        const source = data.source === 'recorded' ? 'Recorded' : matched ? 'Measured frame' : 'Latest detection';
        byId('inspectStatus').textContent = `${source} · ${!left && !right ? 'No board detected' : !shared ? 'No shared IDs' : 'Board detected'}`;
        byId('inspectStatus').title = `Frame ${data.frame_id}; ${matched ? 'counts and overlay match this image' : 'select Detected frame to see its exact image and points'}.`;
        byId('liveAlignment').textContent = px(data.alignment?.p95_px);
        byId('alignmentScope').textContent = `Original pixels · ${shared} shared IDs${state.has_model && !state.candidate ? ' · provisional fit' : ''}`;
        byId('liveRectified').textContent = px(data.rectified?.in_view.p95_px);
        byId('rectifiedScope').textContent = data.rectified
            ? `${data.rectified.in_view.count} / ${shared} shared IDs in view · all points: ${px(data.rectified.all.p95_px)}`
            : 'Inside both rectified images';
        byId('inspectionNote').textContent = data.error ? `Alignment unavailable: ${data.error}`
            : !shared ? 'Show the board to both cameras.'
            : !state.has_model ? 'Detection only. Fit a calibration to measure alignment.'
            : 'Hold still. Live measurements do not change the saved calibration result.';
        this.draw();
    }

    clear(message) {
        this.current = null;
        this.boardKey = null;
        for (const id of ['inspectLeft', 'inspectShared', 'inspectRight', 'liveAlignment', 'liveRectified']) byId(id).textContent = '—';
        byId('inspectStatus').textContent = message;
        byId('alignmentScope').textContent = 'Original pixels · all shared IDs';
        byId('rectifiedScope').textContent = 'Inside both rectified images';
        byId('inspectionNote').textContent = message;
        byId(this.overlayId).replaceChildren();
        byId('detectionBoard').replaceChildren();
    }

    draw() {
        const overlay = byId(this.overlayId);
        overlay.replaceChildren();
        if (!this.current) return;
        const {state, mode, data, matched} = this.current;
        const {width, height} = state.camera;
        const eyeWidth = width / 2;
        const shared = new Set(data.shared_ids);
        overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
        for (const [side, eye] of (matched ? ['left', 'right'] : []).entries()) {
            for (const point of data.points[eye]) {
                const xy = point[mode];
                if (!xy || !xy.every(Number.isFinite)) continue;
                const [x, y] = xy;
                if (x < 0 || y < 0 || x >= eyeWidth || y >= height) continue;
                const fill = colour[shared.has(point.id) ? 'shared' : eye];
                overlay.append(element('circle', {cx: x + side * eyeWidth, cy: y, r: this.pointRadius, fill, stroke: '#18251f', 'stroke-width': 1.5}));
                if (this.panel.open && byId('showPointIds').checked) {
                    const text = element('text', {x: x + side * eyeWidth + 8, y: y - 7, fill, 'font-size': Math.max(16, width / 150), stroke: '#142319', 'stroke-width': 3, 'paint-order': 'stroke', 'font-weight': 600});
                    text.textContent = point.id;
                    overlay.append(text);
                }
            }
        }
        if (!this.panel.open) return;
        const board = byId('detectionBoard');
        const {columns, rows} = state.board;
        const boardKey = JSON.stringify(state.board);
        if (this.boardKey !== boardKey) {
            this.boardKey = boardKey;
            this.boardDots = [];
            board.replaceChildren();
            board.setAttribute('viewBox', `0 0 ${columns * 30} ${rows * 30}`);
            board.append(element('image', {href: `${this.boardUrl}?pattern_only=true&revision=${encodeURIComponent(boardKey)}`, width: columns * 30, height: rows * 30, opacity: .16}));
            for (let id = 0; id < data.total; id++) {
                const x = (1 + id % (columns - 1)) * 30, y = (1 + Math.floor(id / (columns - 1))) * 30;
                const dot = element('circle', {cx: x, cy: y, r: 8, stroke: '#ffffff', 'stroke-width': 1});
                const title = element('title', {});
                dot.append(title);
                const label = element('text', {x, y: y + 3, 'text-anchor': 'middle', 'font-size': 8, fill: '#142319'});
                label.textContent = id;
                board.append(dot, label);
                this.boardDots.push({dot, title});
            }
        }
        board.setAttribute('aria-label', `${columns} by ${rows} squares: ${data.points.left.length} left, ${data.points.right.length} right, ${shared.size} shared corners out of ${data.total}`);
        const left = new Set(data.points.left.map(p => p.id));
        const right = new Set(data.points.right.map(p => p.id));
        for (let id = 0; id < data.total; id++) {
            const status = shared.has(id) ? 'shared' : left.has(id) ? 'left' : right.has(id) ? 'right' : 'missing';
            const {dot, title} = this.boardDots[id];
            dot.setAttribute('fill', colour[status]);
            title.textContent = `Corner ${id}: ${status === 'shared' ? 'both cameras' : status}`;
        }
    }
}
