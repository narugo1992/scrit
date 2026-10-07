import os
import re
from typing import Optional
from urllib.parse import urlsplit, urlencode, parse_qsl

from .common import host_of, segments, fetch_file
from ..errors import ResourceGone, NoContent, ResourceTransient, UnexpectedResponse
from ..http import Fetcher

NAME = 'twitter'

_TWEET_HOSTS = {'x.com', 'twitter.com', 'mobile.twitter.com', 'fxtwitter.com', 'vxtwitter.com', 'fixupx.com'}


def _status_id(url: str) -> Optional[str]:
    if host_of(url) not in _TWEET_HOSTS:
        return None
    matching = re.search(r'/status(?:es)?/(\d+)', urlsplit(url).path)
    return matching.group(1) if matching else None


def _media_id(url: str) -> Optional[str]:
    if host_of(url) != 'pbs.twimg.com':
        return None
    segs = segments(url)
    if len(segs) == 2 and segs[0] == 'media':
        return os.path.splitext(segs[1])[0]
    return None


def match(url: str) -> Optional[str]:
    status = _status_id(url)
    if status:
        return f'twitter_{status}'
    media = _media_id(url)
    return f'twimg_{media}' if media else None


def original_media_url(url: str) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query))
    path, ext = os.path.splitext(parsed.path)
    query.setdefault('format', (ext or '.jpg')[1:])
    query['name'] = 'orig'
    return f'https://pbs.twimg.com{path}?{urlencode(query)}'


def download(fx: Fetcher, url: str, out_dir: str):
    status = _status_id(url)
    if status is None:
        fetch_file(fx, original_media_url(url), out_dir, f'{_media_id(url)}.jpg')
        return

    resp = fx.get(f'https://api.fxtwitter.com/i/status/{status}')
    if resp.status_code == 404:
        raise ResourceGone(f'tweet {status} not found')
    if resp.status_code != 200:
        raise ResourceTransient(f'fxtwitter {status} -> HTTP {resp.status_code}')
    media = (((resp.json().get('tweet') or {}).get('media') or {}).get('all')) or []
    if not media:
        raise NoContent(f'tweet {status} has no media')
    for index, item in enumerate(media, start=1):
        media_url = item.get('url') or ''
        if 'pbs.twimg.com/media/' in media_url:
            media_url = original_media_url(media_url)
            ext = dict(parse_qsl(urlsplit(media_url).query)).get('format', 'jpg')
        else:
            ext = 'mp4'
        try:
            fetch_file(fx, media_url, out_dir, f'{status}_{index}.{ext}')
        except UnexpectedResponse as err:
            raise ResourceTransient(str(err)) from err
