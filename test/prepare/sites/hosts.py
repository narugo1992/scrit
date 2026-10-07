"""Small image / file hosts whose public pages or APIs expose original files without a login."""
import html
import json
import os
import re
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

from .common import host_of, segments, fetch_file, short_hash, clean_id
from ..errors import ResourceGone, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher


class _Site:
    NAME = ''

    @staticmethod
    def match(url: str) -> Optional[str]:
        raise NotImplementedError

    @staticmethod
    def download(fx: Fetcher, url: str, out_dir: str):
        raise NotImplementedError


def _get_text(fx: Fetcher, url: str, **kwargs) -> str:
    resp = fx.get(url, **kwargs)
    if resp.status_code in (404, 410):
        raise ResourceGone(f'{url!r} -> HTTP {resp.status_code}')
    if resp.status_code != 200:
        raise ResourceTransient(f'{url!r} -> HTTP {resp.status_code}')
    return resp.text


def _fetch_all(fx: Fetcher, urls: List[str], out_dir: str, stem: str, ext_hint: str = ''):
    if not urls:
        raise NoContent(f'{stem}: nothing to download')
    for media_url in urls:
        try:
            fetch_file(fx, media_url, out_dir)
        except UnexpectedResponse as err:
            raise ResourceTransient(str(err)) from err


class catbox(_Site):
    NAME = 'catbox'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host == 'files.catbox.moe' and len(segs) == 1:
            return f'catbox_{clean_id(os.path.splitext(segs[0])[0])}'
        if host == 'catbox.moe' and len(segs) == 2 and segs[0] == 'c':
            return f'catbox_album_{clean_id(segs[1])}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        if host_of(url) == 'files.catbox.moe':
            return _fetch_all(fx, [url], out_dir, 'file')
        page = _get_text(fx, url)
        files = sorted(set(re.findall(r'https://files\.catbox\.moe/[\w-]+\.\w+', page)))
        _fetch_all(fx, files[:300], out_dir, 'file')


class imgchest(_Site):
    NAME = 'imgchest'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host == 'imgchest.com' and len(segs) == 2 and segs[0] == 'p':
            return f'imgchest_{clean_id(segs[1])}'
        if host == 'cdn.imgchest.com' and segs:
            return f'imgchest_file_{clean_id(os.path.splitext(segs[-1])[0])}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        if host_of(url) == 'cdn.imgchest.com':
            return _fetch_all(fx, [url], out_dir, 'file')
        page = _get_text(fx, url)
        found = re.search(r'data-page="([^"]+)"', page)
        if not found:
            raise ResourceGone(f'imgchest page {url!r} has no post data')
        post = ((json.loads(html.unescape(found.group(1))).get('props') or {}).get('post')) or {}
        _fetch_all(fx, [item['link'] for item in post.get('files') or []][:300], out_dir, 'file')


class gyazo(_Site):
    NAME = 'gyazo'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host in ('gyazo.com', 'i.gyazo.com') and len(segs) == 1:
            ident = os.path.splitext(segs[0])[0]
            if re.fullmatch(r'[0-9a-f]{32}', ident):
                return f'gyazo_{ident}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        if host_of(url) == 'i.gyazo.com':
            direct = url
        else:
            resp = fx.get('https://api.gyazo.com/api/oembed', params={'url': url})
            direct = (resp.json() if resp.status_code == 200 else {}).get('url')
            if not direct:
                raise ResourceGone(f'gyazo {url!r} has no image')
        try:
            fetch_file(fx, direct, out_dir, gone_statuses=(404, 410, 503))
        except UnexpectedResponse as err:
            raise ResourceGone(str(err)) from err


_IBB_RESERVED = {'album', 'tos', 'login', 'signup', 'page', 'explore', 'privacy', 'plugin', 'upload', 'faq'}


def _og_image(page: str) -> Optional[str]:
    found = re.search(r'<meta property="og:image" content="([^"]+)"', page)
    return html.unescape(found.group(1)) if found else None


class ibb(_Site):
    NAME = 'ibb'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host == 'ibb.co' and len(segs) == 2 and segs[0] == 'album':
            return f'ibb_album_{clean_id(segs[1])}'
        if host == 'ibb.co' and len(segs) == 1 and re.fullmatch(r'[A-Za-z0-9]{6,10}', segs[0]) \
                and segs[0] not in _IBB_RESERVED:
            return f'ibb_{segs[0]}'
        if host == 'i.ibb.co' and len(segs) >= 1:
            return f'ibb_img_{clean_id(segs[0])}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        host = host_of(url)
        if host == 'i.ibb.co':
            return _fetch_all(fx, [url], out_dir, 'file')
        if segments(url)[0] == 'album':
            page = _get_text(fx, url)
            ids = sorted(set(re.findall(r'https://ibb\.co/([A-Za-z0-9]{6,10})"', page)) - _IBB_RESERVED)[:100]
            images = [_og_image(_get_text(fx, f'https://ibb.co/{item}')) for item in ids]
            return _fetch_all(fx, [item for item in images if item], out_dir, 'file')
        image = _og_image(_get_text(fx, url))
        if not image:
            raise ResourceGone(f'ibb page {url!r} has no image')
        _fetch_all(fx, [image], out_dir, 'file')


class postimg(_Site):
    NAME = 'postimg'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host == 'postimg.cc' and len(segs) == 2 and segs[0] == 'gallery':
            return f'postimg_gallery_{clean_id(segs[1])}'
        if host == 'postimg.cc' and len(segs) == 1 and re.fullmatch(r'[A-Za-z0-9]{6,10}', segs[0]):
            return f'postimg_{segs[0]}'
        if host == 'i.postimg.cc' and len(segs) >= 1:
            return f'postimg_img_{clean_id(segs[0])}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        host, segs = host_of(url), segments(url)
        if host == 'i.postimg.cc':
            return _fetch_all(fx, [f'{url.split("?")[0]}?dl=1'], out_dir, 'file')
        if segs[0] == 'gallery':
            listing = fx.post('https://postimg.cc/json', data={'action': 'list', 'album': segs[1], 'page': 1})
            images = listing.json().get('images') or [] if listing.status_code == 200 else []
            pages = [f'https://postimg.cc/{item[0] if isinstance(item, list) else item.get("id")}'
                     for item in images[:100]]
        else:
            pages = [url]
        links = []
        for page in pages:
            found = re.findall(r'href="([^"]+\?dl=1)"', _get_text(fx, page))
            if found:
                links.append(html.unescape(found[0]))
        if not links:
            raise ResourceGone(f'postimg {url!r} has no download link')
        _fetch_all(fx, links, out_dir, 'file')


class gphotos(_Site):
    NAME = 'gphotos'

    @staticmethod
    def match(url):
        host, segs = host_of(url), segments(url)
        if host == 'photos.app.goo.gl' and len(segs) == 1:
            return f'gphotos_{clean_id(segs[0])}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        page = _get_text(fx, url)
        bases = []
        for item in re.findall(r'"(https://lh3\.googleusercontent\.com/pw/[^"=]+)', page):
            if item not in bases:
                bases.append(item)
        if not bases:
            raise NoContent(f'google photos share {url!r} exposes no media')
        _fetch_all(fx, [f'{item}=d' for item in bases[:300]], out_dir, 'photo', '.jpg')


_DIRECT_HOSTS = {'i.pinimg.com', 'file.garden', 'hstorage.io', 'ul.h3z.jp', 'free.picui.cn',
                 'livedoor.blogimg.jp', 'f2.toyhou.se'}


class direct(_Site):
    NAME = 'direct'

    @staticmethod
    def match(url):
        host = host_of(url)
        if host in _DIRECT_HOSTS or host == 'static.wikia.nocookie.net':
            path = urlsplit(url).path
            if os.path.splitext(path)[1].lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.zip', '.psd', '.mp4'} \
                    or host == 'static.wikia.nocookie.net':
                return f'direct_{clean_id(host)}_{short_hash(host + path, 12)}'
        return None

    @staticmethod
    def download(fx, url, out_dir):
        host = host_of(url)
        if host == 'i.pinimg.com':
            url = re.sub(r'pinimg\.com/\d+x/', 'pinimg.com/originals/', url)
        if host == 'static.wikia.nocookie.net':
            # the CDN rejects browser-like user agents, and needs format=original to skip the webp copy
            parsed = urlsplit(url)
            query = dict(parse_qsl(parsed.query))
            query['format'] = 'original'
            return _fetch_all_with_ua(fx, urlunsplit(parsed._replace(query=urlencode(query))), out_dir)
        _fetch_all(fx, [url], out_dir, 'file')


def _fetch_all_with_ua(fx: Fetcher, url: str, out_dir: str):
    try:
        fetch_file(fx, url, out_dir, headers={'User-Agent': 'python-requests/2.32.3'})
    except UnexpectedResponse as err:
        raise ResourceTransient(str(err)) from err
