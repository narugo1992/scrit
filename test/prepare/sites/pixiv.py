import re
from typing import List, Optional
from urllib.parse import urlsplit, parse_qs

from .common import host_of, fetch_file
from ..errors import ResourceGone, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher

NAME = 'pixiv'

_REFERER = {'Referer': 'https://www.pixiv.net/'}
_MAX_PAGES = 50


def _illust_id(url: str) -> Optional[str]:
    if host_of(url) != 'pixiv.net':
        return None
    parsed = urlsplit(url)
    matching = re.search(r'/artworks/(\d+)', parsed.path)
    if matching:
        return matching.group(1)
    if parsed.path.endswith('member_illust.php'):
        return (parse_qs(parsed.query).get('illust_id') or [None])[0]
    return None


def match(url: str) -> Optional[str]:
    illust = _illust_id(url)
    return f'pixiv_{illust}' if illust else None


def _ajax(fx: Fetcher, path: str) -> dict:
    resp = fx.get(f'https://www.pixiv.net/ajax/illust/{path}', headers=_REFERER)
    if resp.status_code in (404, 410):
        raise ResourceGone(f'pixiv {path} not found')
    if resp.status_code != 200:
        raise ResourceTransient(f'pixiv {path} -> HTTP {resp.status_code}')
    return resp.json()


def _original_urls(fx: Fetcher, illust: str) -> List[str]:
    pages = _ajax(fx, f'{illust}/pages')
    if not pages.get('error'):
        urls = [(item.get('urls') or {}).get('original') for item in pages.get('body') or []]
        urls = [item for item in urls if item]
        if urls:
            return urls[:_MAX_PAGES]

    info = _ajax(fx, illust)
    if info.get('error'):
        raise ResourceGone(f'pixiv illust {illust} is gone: {info.get("message")!r}')
    # Logged-out requests get no original URLs for R-18 works. The thumbnail still carries the upload
    # timestamp path, and i.pximg.net only checks the Referer, so rebuild the original URL from it.
    body = info.get('body') or {}
    thumbnail = ((body.get('userIllusts') or {}).get(illust) or {}).get('url') or ''
    matching = re.search(r'/img/(\d+/\d+/\d+/\d+/\d+/\d+)/', thumbnail)
    if not matching:
        raise NoContent(f'pixiv illust {illust}: no original url and no thumbnail path to rebuild it')
    count = min(int(body.get('pageCount') or 1), _MAX_PAGES)
    base = f'https://i.pximg.net/img-original/img/{matching.group(1)}/{illust}_p'
    for ext in ('jpg', 'png', 'gif'):
        if fx.head(f'{base}0.{ext}', headers=_REFERER).status_code == 200:
            return [f'{base}{page}.{ext}' for page in range(count)]
    raise NoContent(f'pixiv illust {illust}: rebuilt original url is not reachable')


def download(fx: Fetcher, url: str, out_dir: str):
    illust = _illust_id(url)
    for index, original in enumerate(_original_urls(fx, illust)):
        ext = original.rsplit('.', 1)[-1]
        try:
            fetch_file(fx, original, out_dir, f'{illust}_p{index}.{ext}', headers=_REFERER)
        except UnexpectedResponse as err:
            raise ResourceTransient(str(err)) from err
