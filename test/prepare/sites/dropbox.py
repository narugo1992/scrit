import os
import zipfile
from typing import Optional
from urllib.parse import urlsplit, urlencode, parse_qsl

from hbutils.system import urlsplit as hb_urlsplit

from .common import fetch_file
from ..errors import ResourceGone, UnexpectedResponse
from ..http import Fetcher

NAME = 'dropbox'

_HOSTS = {'dropbox.com', 'www.dropbox.com', 'dl.dropbox.com', 'dl-web.dropbox.com'}
_PREFIXES = ({'scl', 'fi'}, {'scl', 'fo'})


def _valid(segs) -> bool:
    return (len(segs) >= 3 and segs[0] in ('s', 'sh')) or \
        (len(segs) >= 3 and segs[0] == 'scl' and segs[1] in ('fi', 'fo'))


def match(url: str) -> Optional[str]:
    splitted = hb_urlsplit(url)
    if splitted.host not in _HOSTS:
        return None
    segs = [item for item in splitted.path_segments if item]
    if not _valid(segs):
        return None
    # keep the exact id rule of the existing archive so that already archived resources are deduplicated
    return '_'.join(['dropbox', *segs])


def _direct_url(url: str) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query))
    query['dl'] = '1'
    return f'https://www.dropbox.com{parsed.path}?{urlencode(query)}'


def download(fx: Fetcher, url: str, out_dir: str):
    try:
        target = fetch_file(fx, _direct_url(url), out_dir)
    except UnexpectedResponse as err:
        raise ResourceGone(f'dropbox link not downloadable: {err}') from err
    if os.path.splitext(target)[1].lower() == '.zip' and zipfile.is_zipfile(target):
        with zipfile.ZipFile(target, 'r') as zf:
            zf.extractall(out_dir)
        os.remove(target)
