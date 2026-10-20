// Shared stereo / hand-eye movement illustration. Persistent SVG nodes and CSS animation.
const $ = id => document.getElementById(id);
const set = (id, value) => {if ($(id).textContent !== value) $(id).textContent=value;};
let guideBoardKey;
export function updateBoardGraphic(board, boardUrl="/api/board.png") {
    const key = JSON.stringify([boardUrl, board]);
    if (key === guideBoardKey) return;
    guideBoardKey = key;
    const url = `${boardUrl}?pattern_only=true&revision=${encodeURIComponent(key)}`;
    const ratio = board.columns / board.rows;
    const width = Math.min(148, 104 * ratio), height = width / ratio;
    const x = 130 - width / 2, y = 76 - height / 2;
    for (const [id, padding] of [['guideBoard', 0], ['boardPaper', 6]]) {
        const element = $(id);
        element.setAttribute('x', x - padding);
        element.setAttribute('y', y - padding);
        element.setAttribute('width', width + padding * 2);
        element.setAttribute('height', height + padding * 2);
    }
    $('guideBoard').setAttribute('href', url);
    $('positionBoard').setAttribute('href', url);
    $('positionBoard').setAttribute('width', 36);
    $('positionBoard').setAttribute('height', 36 / ratio);
    set('guideBoardLabel', `${board.columns} × ${board.rows} · ${board.square_mm} mm squares`);
}

export function updateMovementGuide(adjustment, board, {holding=false, paused=false}={}) {
    const motion = adjustment.kind === 'tilt' ? adjustment.edge
        : adjustment.kind === 'roll' ? adjustment.direction
        : adjustment.kind === 'distance' ? (adjustment.direction === 'closer' ? 'near' : 'far')
        : adjustment.kind === 'move' ? 'position' : 'still';
    $('illustration').dataset.motion = motion;
    $('illustration').classList.toggle('holding', holding || paused);
    const position = adjustment.kind === 'move';
    $('positionDrawing').style.display = position ? '' : 'none';
    $('boardDrawing').style.display = position ? 'none' : '';
    $('boardShadow').style.display = position ? 'none' : '';
    if (position) {
        const [x, y] = adjustment.target;
        const tx = 38 + x * 184, ty = 24 + y * 104;
        const current = adjustment.current;
        const cx = current ? 38 + Math.max(0, Math.min(1, current[0])) * 184 : tx;
        const cy = current ? 24 + Math.max(0, Math.min(1, current[1])) * 104 : ty;
        $('positionTarget').setAttribute('cx', tx);
        $('positionTarget').setAttribute('cy', ty);
        $('positionBoard').setAttribute('x', cx - 18);
        $('positionBoard').setAttribute('y', cy - 18 * board.rows / board.columns);
        $('positionBoard').setAttribute('opacity', current ? 1 : .45);
        const dx = tx - cx, dy = ty - cy;
        const distance = Math.hypot(dx, dy);
        $('positionArrow').style.display = current && distance > 28 ? '' : 'none';
        if (distance > 28) {
            $('positionArrow').setAttribute('x1', cx + 18 * dx / distance);
            $('positionArrow').setAttribute('y1', cy + 18 * dy / distance);
            $('positionArrow').setAttribute('x2', tx - 12 * dx / distance);
            $('positionArrow').setAttribute('y2', ty - 12 * dy / distance);
        }
    }
    const boardWidth = Number($('guideBoard').getAttribute('width'));
    const boardHeight = Number($('guideBoard').getAttribute('height'));
    const bx = 130 - boardWidth / 2, by = 76 - boardHeight / 2;
    const edges = {
        left: `M${bx - 4} ${by}v${boardHeight}`,
        right: `M${bx + boardWidth + 4} ${by}v${boardHeight}`,
        top: `M${bx} ${by - 4}h${boardWidth}`,
        bottom: `M${bx} ${by + boardHeight + 4}h${boardWidth}`
    };
    $('edgeHighlight').setAttribute('d', edges[motion] || '');
}
