import os
from contextlib import contextmanager

import pyrfc6266
import requests
from tqdm.auto import tqdm

from .session import srequest, get_requests_session


class _FakeClass:
    def update(self, *args, **kwargs):
        pass


@contextmanager
def _with_tqdm(expected_size, desc, silent: bool = False):
    if not silent:
        with tqdm(total=expected_size, unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            yield pbar
    else:
        yield _FakeClass()


def download_file(url, filename=None, output_directory=None,
                  expected_size: int = None, desc=None, session=None, silent: bool = False,
                  **kwargs):
    session = session or get_requests_session()
    response = srequest(session, 'GET', url, stream=True, allow_redirects=True, **kwargs)
    expected_size = expected_size or response.headers.get('Content-Length', None)
    if filename is None:
        diso = response.headers.get('Content-Disposition')
        if not diso:
            raise RuntimeError('Filename not given, and it cannot be determined in the response headers.')
        filename = pyrfc6266.parse_filename(diso)
    if output_directory is not None:
        filename = os.path.join(output_directory, filename)
    expected_size = int(expected_size) if expected_size is not None else expected_size

    desc = desc or os.path.basename(filename)
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'wb') as f:
        with _with_tqdm(expected_size, desc, silent) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))

    actual_size = os.path.getsize(filename)
    if expected_size is not None and actual_size != expected_size:
        os.remove(filename)
        raise requests.exceptions.HTTPError(f"Downloaded file is not of expected size, "
                                            f"{expected_size} expected but {actual_size} found.")

    return filename
