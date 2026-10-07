import os
import re
import zipfile
from typing import Set

from hbutils.system import TemporaryDirectory

from .errors import NoContent
from .http import Fetcher


def write_zip(src_dir: str, zip_file: str, prefix: str = '') -> int:
    """Flatten ``src_dir`` into ``zip_file``.

    Every file is stored as ``prefix + sanitized(relative path without extension) + extension``, which is
    the layout downstream consumers already depend on. Returns the number of files written.
    """
    used: Set[str] = set()
    written = 0
    with zipfile.ZipFile(zip_file, 'w') as zf:
        for root, _, files in sorted(os.walk(src_dir)):
            for file in sorted(files):
                if file.startswith('.part_'):
                    continue
                filename = os.path.join(root, file)
                relname_body, relname_ext = os.path.splitext(os.path.relpath(filename, src_dir))
                body = re.sub(r'[\W_]+', '_', relname_body).strip('_') or 'file'
                arcname = f'{prefix}{body}{relname_ext}'
                counter = 1
                while arcname in used:
                    counter += 1
                    arcname = f'{prefix}{body}_{counter}{relname_ext}'
                used.add(arcname)
                zf.write(filename, arcname)
                written += 1
    return written


def build_zip(site, url: str, prefix: str, zip_file: str, fx: Fetcher):
    """Download ``url`` with ``site`` and pack it into ``zip_file``; raise ``NoContent`` for empty results."""
    with TemporaryDirectory() as td:
        site.download(fx, url, td)
        if write_zip(td, zip_file, prefix) == 0:
            os.remove(zip_file)
            raise NoContent(f'{url!r} produced no file')
