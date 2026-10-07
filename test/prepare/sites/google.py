import logging
import mimetypes
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

from gdown.download_folder import _parse_google_drive_file

from .common import host_of, fetch_file
from ..errors import ResourceGone, ResourceBlocked, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher, safe_name

NAME = 'google'

_ID = r'[A-Za-z0-9_-]{10,}'
_FOLDER_MIME = 'application/vnd.google-apps.folder'
_IMAGE_VIA_LH3 = {'image/png', 'image/jpeg', 'image/gif', 'image/webp'}
_DOC_EXPORT = {
    'application/vnd.google-apps.document': ('document', 'docx'),
    'application/vnd.google-apps.spreadsheet': ('spreadsheets', 'xlsx'),
    'application/vnd.google-apps.presentation': ('presentation', 'pptx'),
}
_MAX_DRIVE_API_DEPTH = 5


def parse(url: str) -> Optional[Tuple[str, str]]:
    """Return ``(kind, id)`` where kind is folder / file / document / spreadsheets / presentation."""
    host = host_of(url)
    if host not in {'drive.google.com', 'docs.google.com', 'drive.usercontent.google.com'}:
        return None
    parsed = urlsplit(url)
    for pattern, kind in [
        (rf'/folders/({_ID})', 'folder'),
        (rf'/file/d/({_ID})', 'file'),
        (rf'/(document|spreadsheets|presentation)/d/({_ID})', None),
    ]:
        matching = re.search(pattern, parsed.path)
        if matching:
            if kind is None:
                return matching.group(1), matching.group(2)
            return kind, matching.group(1)
    matching = re.search(rf'(?:^|&)id=({_ID})(?:&|$)', parsed.query)
    if matching and parsed.path.rstrip('/') in {'/open', '/uc', '/download'}:
        return 'file', matching.group(1)
    return None


def match(url: str) -> Optional[str]:
    parsed = parse(url)
    return f'googledrive_{parsed[1]}' if parsed else None


def download(fx: Fetcher, url: str, out_dir: str):
    kind, ident = parse(url)
    if kind == 'folder':
        entries = _list_folder(fx, ident)
        if not entries:
            raise NoContent(f'Drive folder {ident} is empty')
        for file_id, parts, mime in entries:
            _download_entry(fx, file_id, os.path.join(out_dir, *parts[:-1]), parts[-1], mime)
    elif kind == 'file':
        _download_entry(fx, ident, out_dir, None, None)
    else:
        _download_entry(fx, ident, out_dir, f'{ident}', f'application/vnd.google-apps.{kind.rstrip("s")}')


# ---------------------------------------------------------------- listing
def _fix_name(name: str, mime: str) -> str:
    name = safe_name(name)
    ext = os.path.splitext(name)[1]
    guesses = [item.lower() for item in mimetypes.guess_all_extensions(mime)]
    if not ext or (guesses and ext.lower() not in guesses):
        name += mimetypes.guess_extension(mime) or ''
    return name


_MAX_FOLDER_PAGES = 200
_PAGE_LIMIT = 50  # a Drive folder page lists at most this many direct children


def _parse_page(fx: Fetcher, folder_id: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Fetch one folder page; returns the folder name and its ``(id, name, mime)`` children."""
    url = f'https://drive.google.com/drive/folders/{folder_id}?hl=en'
    resp = fx.get(url)
    if resp.status_code == 429 or 'google.com/sorry' in resp.url:
        raise ResourceBlocked(f'Drive folder page {folder_id} is throttled', cooldown=10 * 60.0)
    if resp.status_code in (404, 410) or 'accounts.google.com' in resp.url:
        raise ResourceGone(f'Drive folder {folder_id} is gone or private (HTTP {resp.status_code})')
    if resp.status_code != 200:
        raise ResourceTransient(f'Drive folder page {folder_id} -> HTTP {resp.status_code}')
    try:
        node, children = _parse_google_drive_file(url, resp.text)
    except RuntimeError as err:
        # gdown raises RuntimeError for a normal page without folder data: not shared, or no longer there
        raise ResourceGone(f'Drive folder {folder_id} cannot be read: {err}') from err
    return node.name, list(children)


def _list_folder(fx: Fetcher, folder_id: str) -> List[Tuple[str, List[str], str]]:
    entries: List[Tuple[str, List[str], str]] = []
    truncated = []
    pages = [0]
    root_name = ['']

    def walk(fid: str, parts: List[str], depth: int):
        pages[0] += 1
        name, children = _parse_page(fx, fid)
        if fid == folder_id:
            root_name[0] = name
        base = [*parts, safe_name(name)]
        if len(children) >= _PAGE_LIMIT:
            truncated.append(fid)
        for child_id, child_name, child_type in children:
            if child_type == _FOLDER_MIME:
                if depth < _MAX_DRIVE_API_DEPTH and pages[0] < _MAX_FOLDER_PAGES:
                    walk(child_id, base, depth + 1)
            else:
                entries.append((child_id, [*base, _fix_name(child_name, child_type)], child_type))

    walk(folder_id, [], 0)
    if truncated:
        logging.info(f'Drive folder {folder_id} has a folder with >= {_PAGE_LIMIT} children, listing via the Drive API')
        api_entries = _list_folder_api(fx, folder_id, root_name[0])
        if len(api_entries) >= len(entries):
            return api_entries
        logging.warning(f'Drive API listing for {folder_id} is shorter ({len(api_entries)}) than the page '
                        f'listing ({len(entries)}), keep the page listing.')
    return entries


def _api_key(fx: Fetcher, folder_id: str) -> Optional[str]:
    cached = getattr(fx, '_drive_api_key', None)
    if cached:
        return cached
    page = fx.get(f'https://drive.google.com/drive/folders/{folder_id}').text
    for key in sorted(set(re.findall(r'AIza[0-9A-Za-z_\-]{35}', page))):
        resp = fx.get('https://www.googleapis.com/drive/v3/files',
                      params={'key': key, 'q': f"'{folder_id}' in parents", 'pageSize': 1},
                      headers={'Referer': 'https://drive.google.com/'})
        if resp.status_code == 200:
            fx._drive_api_key = key
            return key
    return None


def _list_folder_api(fx: Fetcher, folder_id: str, root_name: str) -> List[Tuple[str, List[str], str]]:
    key = _api_key(fx, folder_id)
    if key is None:
        raise ResourceBlocked(f'no usable Drive API key to list folder {folder_id}', cooldown=10 * 60.0)

    entries: List[Tuple[str, List[str], str]] = []

    def walk(fid: str, parts: List[str], depth: int):
        token = None
        while True:
            params = {'key': key, 'q': f"'{fid}' in parents and trashed=false", 'pageSize': 1000,
                      'fields': 'nextPageToken,files(id,name,mimeType)'}
            if token:
                params['pageToken'] = token
            resp = fx.get('https://www.googleapis.com/drive/v3/files', params=params,
                          headers={'Referer': 'https://drive.google.com/'})
            if resp.status_code == 403:
                raise ResourceBlocked(f'Drive API listing refused for {fid}', cooldown=10 * 60.0)
            if resp.status_code != 200:
                raise ResourceTransient(f'Drive API listing for {fid} -> HTTP {resp.status_code}')
            body = resp.json()
            for item in body.get('files') or []:
                if item['mimeType'] == _FOLDER_MIME:
                    if depth < _MAX_DRIVE_API_DEPTH:
                        walk(item['id'], [*parts, safe_name(item['name'])], depth + 1)
                else:
                    entries.append((item['id'], [*parts, _fix_name(item['name'], item['mimeType'])], item['mimeType']))
            token = body.get('nextPageToken')
            if not token:
                return

    walk(folder_id, [safe_name(root_name)], 0)
    return entries


# ---------------------------------------------------------------- downloading
def _classify_html(err: UnexpectedResponse, what: str):
    text = err.snippet.lower()
    if any(word in text for word in ('too many users', 'many accesses', 'quota exceeded', 'unusual traffic')):
        raise ResourceBlocked(f'{what}: download quota exceeded') from err
    if err.status in (401, 403, 404) or any(word in text for word in (
            'request access', 'you need access', 'sign in', 'not found', 'permission')):
        raise ResourceGone(f'{what}: not accessible (HTTP {err.status})') from err
    raise ResourceTransient(f'{what}: unexpected response {err}') from err


def _download_entry(fx: Fetcher, file_id: str, out_dir: str, name: Optional[str], mime: Optional[str]):
    os.makedirs(out_dir, exist_ok=True)
    if mime in _DOC_EXPORT:
        kind, fmt = _DOC_EXPORT[mime]
        url = f'https://docs.google.com/{kind}/d/{file_id}/export?format={fmt}'
        try:
            fetch_file(fx, url, out_dir, f'{os.path.splitext(name)[0]}.{fmt}')
        except UnexpectedResponse as err:
            _classify_html(err, f'export of {file_id}')
        return
    if mime == 'application/vnd.google-apps.shortcut':
        logging.warning(f'Drive shortcut {file_id} ({name!r}) skipped.')
        return

    endpoints = []
    if mime in _IMAGE_VIA_LH3:
        endpoints.append(f'https://lh3.googleusercontent.com/d/{file_id}=d')
    endpoints.append(f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t')

    last_error: Optional[UnexpectedResponse] = None
    for url in endpoints:
        try:
            fetch_file(fx, url, out_dir, name, gone_statuses=())
        except UnexpectedResponse as err:
            last_error = err
            continue
        return
    _classify_html(last_error, f'file {file_id}')
