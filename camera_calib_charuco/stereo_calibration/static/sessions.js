/** Session navigation; capture workflows and preview rendering live in app.js. */
export class SessionPicker {
    constructor({request, getState, onSwitch}) {
        this.request = request;
        this.getState = getState;
        this.onSwitch = onSwitch;
        this.dialog = document.getElementById('sessionDialog');
        this.list = document.getElementById('sessionList');
        this.error = document.getElementById('sessionError');
        this.busy = false;
        this.initialPicker = false;
        document.getElementById('sessionsButton').addEventListener('click', () => this.open());
        document.getElementById('chooseSession').addEventListener('click', () => this.open());
        document.getElementById('closeSessions').addEventListener('click', () => this.dialog.close());
        document.getElementById('newSession').addEventListener('click', () => this.select(null));
        document.getElementById('openSessionsFolder').addEventListener('click', () => this.openFolder('sessions'));
        document.getElementById('sessionPathForm').addEventListener('submit', event => {
            event.preventDefault();
            const path = document.getElementById('openSessionPath').value.trim();
            if (path) this.select(path);
        });
        this.dialog.addEventListener('cancel', event => {
            if (this.busy) event.preventDefault();
        });
    }

    update(state) {
        document.getElementById('activeSession').textContent = state.session
            ? state.session.split('/').pop() : state.phase === 'picker' ? '' : 'New session';
        document.getElementById('activeSession').title = state.session || '';
        if (state.phase === 'picker' && !this.initialPicker) {
            this.initialPicker = true;
            this.open();
        }
    }

    async open() {
        if (this.dialog.open) return;
        this.dialog.showModal();
        this.error.hidden = true;
        this.list.textContent = 'Loading sessions…';
        try {
            const state = this.getState();
            if (state && !state.paused && ['collecting', 'validating'].includes(state.phase)) {
                this.onSwitch(await this.request('/api/control', {action: 'pause'}));
            }
            const data = await this.request('/api/sessions');
            document.getElementById('sessionDirectory').textContent = data.output;
            this.render(data.sessions);
        } catch (error) {
            this.list.textContent = '';
            this.showError(error);
        }
    }

    showError(error) {
        this.error.textContent = error.message;
        this.error.hidden = false;
    }

    async openFolder(kind, path) {
        this.error.hidden = true;
        try {
            await this.request('/api/folders/open', {kind, path});
        } catch (error) {
            this.showError(error);
        }
    }

    render(sessions) {
        this.list.replaceChildren();
        if (!sessions.length) {
            this.list.textContent = 'No saved sessions yet.';
            return;
        }
        for (const session of sessions) {
            const row = document.createElement('article');
            row.className = 'session-row';
            const info = document.createElement('div');
            const title = document.createElement('strong');
            const date = new Date(session.created_at);
            title.textContent = Number.isNaN(date.valueOf()) ? session.name
                : date.toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'});
            title.title = session.path;
            const details = document.createElement('p');
            details.textContent = session.error || `${session.captures} pairs · ${session.camera.width} × ${session.camera.height} · ${session.board.square_mm} mm squares`;
            const status = document.createElement('span');
            status.className = 'badge';
            status.textContent = session.error ? 'Unavailable' : session.passed ? 'Checks passed'
                : session.phase === 'ready' ? 'Not started' : 'In progress';
            info.append(title, details, status);
            const actions = document.createElement('div');
            actions.className = 'session-actions';
            for (const [label, replay] of [['Open', false], ['Replay', true]]) {
                const button = document.createElement('button');
                button.textContent = label;
                button.disabled = !!session.error || (replay && !session.captures);
                button.addEventListener('click', () => this.select(session.path, replay));
                actions.append(button);
            }
            const folder = document.createElement('button');
            folder.textContent = 'Folder';
            folder.title = `Open in file manager: ${session.path}`;
            folder.addEventListener('click', () => this.openFolder('session', session.path));
            actions.append(folder);
            row.append(info, actions);
            this.list.append(row);
        }
    }

    async select(path, replay = false) {
        if (this.busy) return;
        this.busy = true;
        this.error.hidden = true;
        const controls = [...this.dialog.querySelectorAll('button, input')];
        const disabled = controls.map(control => control.disabled);
        controls.forEach(control => control.disabled = true);
        this.dialog.setAttribute('aria-busy', 'true');
        try {
            const state = await this.request('/api/sessions/open', {path, replay});
            this.onSwitch(state);
            this.dialog.close();
        } catch (error) {
            this.showError(error);
        } finally {
            controls.forEach((control, i) => control.disabled = disabled[i]);
            this.dialog.removeAttribute('aria-busy');
            this.busy = false;
        }
    }
}
