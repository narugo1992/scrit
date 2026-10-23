import logging
import os
import shutil
import zipfile
from typing import Optional
from urllib.parse import urlsplit, urlencode, parse_qsl

from hbutils.system import urlsplit as hb_urlsplit

from .common import fetch_file
from ..errors import ResourceGone, UnexpectedResponse
from ..http import Fetcher, safe_name

NAME = 'dropbox'

_HOSTS = {'dropbox.com', 'www.dropbox.com', 'dl.dropbox.com', 'dl-web.dropbox.com'}
_MAX_MEMBERS = 5000
_MAX_EXTRACT_BYTES = 4 * 1024 ** 3


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


def _free_path(path: str) -> str:
    """Names cut to the length limit can collide, so a numbered sibling is used instead of overwriting."""
    stem, ext = os.path.splitext(path)
    index = 1
    while os.path.exists(path):
        path = f'{stem}_{index}{ext}'
        index += 1
    return path


def _extract(zf: zipfile.ZipFile, out_dir: str):
    """Unpack member by member: every path component goes through `safe_name`, which `extractall` does not do."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = [safe_name(part) for part in info.filename.replace('\\', '/').split('/') if part not in ('', '.', '..')]
        if not parts:
            continue
        dest = os.path.join(out_dir, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(info) as src, open(_free_path(dest), 'wb') as dst:
            shutil.copyfileobj(src, dst)


def download(fx: Fetcher, url: str, out_dir: str):
    try:
        target = fetch_file(fx, _direct_url(url), out_dir)
    except UnexpectedResponse as err:
        raise ResourceGone(f'dropbox link not downloadable: {err}') from err
    if os.path.splitext(target)[1].lower() == '.zip' and zipfile.is_zipfile(target):
        with zipfile.ZipFile(target, 'r') as zf:
            infos = zf.infolist()
            if len(infos) > _MAX_MEMBERS or sum(item.file_size for item in infos) > _MAX_EXTRACT_BYTES:
                logging.warning(f'{target!r} is too big to unpack ({len(infos)} members), kept as a zip.')
                return
            _extract(zf, out_dir)
        os.remove(target)
