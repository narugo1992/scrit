import logging
import os.path
import pathlib
from functools import lru_cache

_IP_FILE = os.path.join(os.path.dirname(__file__), 'iplist.txt')


@lru_cache()
def get_proxies():
    retval = []
    for line in pathlib.Path(_IP_FILE).read_text().splitlines(keepends=False):
        line = line.strip()
        if line:
            if line.startswith('#'):
                logging.warning(f'Proxy iplist {line!r} will be skipped.')
            else:
                retval.append(f'http://{line}:80')

    return retval
