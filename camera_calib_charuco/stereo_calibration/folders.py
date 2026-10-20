"""Locate diagnostic directories and open app-owned folders on the local desktop."""

import json
import shutil
import subprocess
from pathlib import Path


def inspection_folders(session, root, current=None):
    """Read only the metadata line; diagnostic frames are never decoded here."""
    if session is None:
        return []
    session = Path(session).resolve()
    candidates = {p.resolve() for base in (session / 'diagnostics', Path(root))
                  for p in base.glob('**/readings.jsonl')}
    if current:
        candidates.add(Path(current) / 'readings.jsonl')
    folders = []
    for file in candidates:
        try:
            with file.open() as stream:
                metadata = json.loads(stream.readline())
            if (not isinstance(metadata, dict) or metadata.get('type') != 'metadata'
                    or not isinstance(metadata.get('utc'), str)):
                continue
            if not file.is_relative_to(session / 'diagnostics') and (
                Path(metadata.get('session', '')).resolve() != session
            ):
                continue
            folders.append({'path': str(file.parent.resolve()), 'created_at': metadata['utc']})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(folders, key=lambda folder: folder['created_at'], reverse=True)


def open_directory(path):
    """Use the desktop's directory handler, with an absolute URI and no shell."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f'This folder does not exist yet: {path}')
    if gio := shutil.which('gio'):
        command = [gio, 'open', path.as_uri()]
    elif opener := shutil.which('xdg-open'):
        command = [opener, path.as_uri()]
    else:
        raise OSError(f'No desktop folder opener is installed. Folder: {path}')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except subprocess.TimeoutExpired as error:
        raise OSError(f'The file manager did not respond. Folder: {path}') from error
    if result.returncode:
        raise OSError(f'Could not open the file manager: {result.stderr.strip() or path}')
    return path
