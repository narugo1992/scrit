import base64
import logging
import os
from typing import Optional
from urllib.parse import urlsplit

from .common import host_of, fetch_file, short_hash
from ..errors import ResourceGone, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher, safe_name

NAME = 'onedrive'

_BADGER_APP_ID = '5cbed6ac-a083-4e14-b191-b4ba07653de2'
_API = 'https://my.microsoftpersonalcontent.com/_api/v2.0'
_MAX_FILE_BYTES = 600 * 1024 ** 2
_MAX_TOTAL_BYTES = 2 * 1024 ** 3
_MAX_DEPTH = 3


def match(url: str) -> Optional[str]:
    if host_of(url) not in ('1drv.ms', 'onedrive.live.com'):
        return None
    parsed = urlsplit(url)
    if len(parsed.path.strip('/')) < 8:
        return None
    return f'onedrive_{short_hash(host_of(url) + parsed.path, 16)}'


def _headers(fx: Fetcher) -> dict:
    token = getattr(fx, '_badger_token', None)
    if token is None:
        resp = fx.post('https://api-badgerp.svc.ms/v1.0/token', json={'appId': _BADGER_APP_ID})
        if resp.status_code != 200:
            raise ResourceTransient(f'badger token -> HTTP {resp.status_code}')
        token = fx._badger_token = resp.json()['token']
    return {'Authorization': f'Badger {token}', 'Prefer': 'autoredeem'}


def _json(fx: Fetcher, url: str, **kwargs) -> dict:
    resp = fx.get(url, headers=_headers(fx), **kwargs)
    if resp.status_code in (400, 401, 403, 404):
        raise ResourceGone(f'onedrive {url!r} -> HTTP {resp.status_code}')
    if resp.status_code != 200:
        raise ResourceTransient(f'onedrive {url!r} -> HTTP {resp.status_code}')
    return resp.json()


def download(fx: Fetcher, url: str, out_dir: str):
    encoded = 'u!' + base64.urlsafe_b64encode(url.encode()).decode().rstrip('=')
    root = _json(fx, f'{_API}/shares/{encoded}/driveitem')
    total = [0]
    saved = [0]

    def save(item: dict, directory: str):
        size = item.get('size') or 0
        link = item.get('@content.downloadUrl')
        if not link:
            return
        if size > _MAX_FILE_BYTES or total[0] + size > _MAX_TOTAL_BYTES:
            logging.warning(f'onedrive file {item.get("name")!r} ({size} bytes) skipped by the size limits.')
            return
        total[0] += size
        try:
            fetch_file(fx, link, directory, safe_name(item.get('name') or 'file'))
        except UnexpectedResponse as err:
            raise ResourceTransient(str(err)) from err
        saved[0] += 1

    def walk(item: dict, directory: str, depth: int):
        if 'folder' not in item:
            save(item, directory)
            return
        if depth > _MAX_DEPTH:
            return
        directory = os.path.join(directory, safe_name(item.get('name') or 'onedrive'))
        children = item.get('children')
        if children is None:
            drive = item['parentReference']['driveId']
            children = _json(fx, f'{_API}/drives/{drive}/items/{item["id"]}/children').get('value') or []
        for child in children:
            walk(child, directory, depth + 1)

    walk(root, out_dir, 0)
    if saved[0] == 0:
        raise NoContent(f'onedrive {url!r}: no downloadable file within the size limits')
