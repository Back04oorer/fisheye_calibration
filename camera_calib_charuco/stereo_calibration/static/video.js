/** Latest-only video delivery, independent of slower calibration state updates. */
export class VideoPreview {
    constructor({getState, onFrame, onError}) {
        this.getState = getState;
        this.onFrame = onFrame;
        this.onError = onError;
        this.image = document.getElementById('frame');
        this.projection = document.getElementById('projection');
        this.source = document.getElementById('previewSource');
        this.revision = 0;
        this.reset();
        for (const control of [this.projection, this.source]) {
            control.addEventListener('change', () => this.reset());
        }
    }

    get matched() {
        return this.getState()?.replay || this.source.value === 'measured';
    }

    reset() {
        this.revision++;
        this.abort?.abort();
        if (this.url) URL.revokeObjectURL(this.url);
        this.url = null;
        this.key = null;
        this.after = -1;
        this.lastReceived = 0;
        this.times = [];
        this.image.hidden = true;
        this.image.removeAttribute('src');
        document.getElementById('pointOverlay').replaceChildren();
    }

    async frame() {
        const s = this.getState();
        if (!s || s.phase === 'picker' || s.error || document.hidden
            || document.getElementById('sessionDialog').open) return;
        const revision = this.revision, mode = this.projection.value, matched = this.matched;
        if (matched && s.frame_id < 0) return;
        const key = `${s.workspace_id}:${s.frame_id}:${mode}`;
        if (matched && key === this.key) return;
        const query = new URLSearchParams({workspace_id: s.workspace_id, mode});
        query.set(matched ? 'frame_id' : 'after', matched ? s.frame_id : this.after);
        this.abort = new AbortController();
        const response = await fetch(`/api/${matched ? 'frame' : 'live'}.jpg?${query}`, {
            signal: this.abort.signal,
        });
        if (response.status === 204 || response.status === 404 || response.status === 409) return;
        if (!response.ok) throw new Error(`Preview unavailable (${response.status})`);
        const url = URL.createObjectURL(await response.blob());
        const decoded = new Image();
        decoded.src = url;
        try {
            await decoded.decode();
        } catch (error) {
            URL.revokeObjectURL(url);
            throw error;
        }
        if (revision !== this.revision || this.getState()?.workspace_id !== s.workspace_id) {
            URL.revokeObjectURL(url);
            return;
        }
        this.image.src = url;
        this.image.hidden = false;
        this.image.alt = matched ? 'Stereo camera measurement with detected points' : 'Live stereo camera';
        this.image.dataset.frameId = response.headers.get('X-Frame-Id');
        this.image.dataset.source = matched ? 'measured' : 'live';
        if (this.url) URL.revokeObjectURL(this.url);
        this.url = url;
        this.after = Number(response.headers.get('X-Frame-Id'));
        this.key = key;
        this.lastReceived = performance.now();
        this.times.push(this.lastReceived);
        this.times = this.times.filter(t => t >= this.lastReceived - 2000);
        const fps = this.times.length > 1
            ? (this.times.length - 1) * 1000 / (this.lastReceived - this.times[0]) : 0;
        document.getElementById('emptyFrame').hidden = true;
        const label = document.getElementById('frameRate');
        label.textContent = s.replay ? 'Recorded frame' : `${fps.toFixed(1)} video · ${s.detection_fps.toFixed(1)} detection fps`;
        label.title = matched ? 'Exact image used for this detection'
            : `${response.headers.get('X-Frame-Age-Ms')} ms from camera receipt to preview encoding`;
        document.getElementById('previewLabel').textContent = matched
            ? 'Measured frame · points match this image' : mode === 'raw'
                ? s.goal?.target ? 'Live video · blue area is the target' : 'Live video'
                : 'Live rectified video';
        this.onFrame(s, mode, matched);
    }

    async run() {
        const start = performance.now();
        try {
            await this.frame();
        } catch (error) {
            if (error.name !== 'AbortError') this.onError(error);
        }
        if (!this.getState()?.replay && this.lastReceived
            && performance.now() - this.lastReceived > 2000) {
            document.getElementById('frameRate').textContent = 'Waiting for video';
        }
        // One request at a time; missed camera frames are skipped, never queued.
        setTimeout(() => this.run(), Math.max(0, 1000 / 30 - (performance.now() - start)));
    }
}
