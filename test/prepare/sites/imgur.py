import mimetypes
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from .common import host_of, segments, fetch_file
from ..errors import ResourceGone, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher, safe_name

NAME = 'imgur'

_API = 'https://api.imgur.com/post/v1'
_DEFAULT_CLIENT_ID = 'd70305e7c3ac5c6'
_RESERVED = {'a', 'gallery', 't', 'user', 'upload', 'signin', 'register', 'search', 'removalrequest', 'blog', 'apps'}
_ID = re.compile(r'^[A-Za-z0-9]{5,8}$')


def _tail(slug: str) -> str:
    return slug.split('-')[-1]


def parse(url: str):
    host = host_of(url)
    segs = segments(url)
    if host == 'imgur.com' and len(segs) >= 2 and segs[0] == 'a':
        return 'album', _tail(segs[1])
    if host == 'imgur.com' and len(segs) >= 2 and segs[0] == 'gallery':
        return 'post', _tail(segs[1])
    if host == 'imgur.com' and len(segs) == 1 and segs[0] not in _RESERVED:
        ident = os.path.splitext(segs[0])[0]
        return ('media', ident) if _ID.match(ident) else None
    if host == 'i.imgur.com' and len(segs) == 1:
        ident = os.path.splitext(segs[0])[0]
        return ('direct', ident) if _ID.match(ident) else None
    return None


def match(url: str) -> Optional[str]:
    parsed = parse(url)
    if parsed is None:
        return None
    kind, ident = parsed
    return f'imgur_{ident}' if kind in ('album', 'post') else f'imgur_media_{ident}'


@lru_cache()
def _scraped_client_id(fx_id: int, fx: Fetcher) -> str:
    html = fx.get('https://imgur.com/').text
    for src in re.findall(r'<script[^>]+src="([^"]*main[^"]*\.js)"', html):
        script = fx.get(src if src.startswith('http') else f'https://imgur.com{src}').text
        found = re.findall(r'apiClientId:\s*"([a-z\d]+)"', script)
        if found:
            return found[0]
    raise ResourceTransient('imgur client id not found in main.js')


def _api(fx: Fetcher, path: str) -> Optional[dict]:
    for client_id in (_DEFAULT_CLIENT_ID, None):
        if client_id is None:
            client_id = _scraped_client_id(id(fx), fx)
        resp = fx.get(f'{_API}/{path}', params={'client_id': client_id, 'include': 'media'},
                      headers={'Referer': 'https://imgur.com/'})
        if resp.status_code in (400, 404):
            return None
        if resp.status_code in (401, 403):
            continue
        if resp.status_code != 200:
            raise ResourceTransient(f'imgur API {path} -> HTTP {resp.status_code}')
        return resp.json()
    raise ResourceTransient(f'imgur API {path} refused both client ids')


def _media_items(body: dict) -> List[Dict]:
    return list(body.get('media') or ([body] if body.get('url') else []))


def download(fx: Fetcher, url: str, out_dir: str):
    kind, ident = parse(url)
    if kind == 'direct':
        direct = f'https://i.imgur.com/{os.path.basename(urlsplit(url).path)}'
        if os.path.splitext(direct)[1]:
            try:
                fetch_file(fx, direct, out_dir, safe_name(os.path.basename(direct)), gone_if_redirected_to=('removed',))
                return
            except UnexpectedResponse as err:
                if err.status != 200:
                    raise ResourceTransient(str(err)) from err
                # a 200 html page: imgur wraps the image, the media API knows the real file
        # i.imgur.com/<id> without an extension is an html wrapper page too
        kind = 'media'
    body = _api(fx, {'album': f'albums/{ident}', 'post': f'posts/{ident}', 'media': f'media/{ident}'}[kind])
    if body is None:
        raise ResourceGone(f'imgur {kind} {ident} not found')
    items = _media_items(body)
    if not items:
        raise NoContent(f'imgur {kind} {ident} has no media')
    for index, item in enumerate(items, start=1):
        name = item.get('name') or f'{ident}_{index}'
        if not os.path.splitext(name)[1]:
            name += mimetypes.guess_extension(item.get('mime_type') or '') or ''
        fetch_file(fx, item['url'], out_dir, name)
