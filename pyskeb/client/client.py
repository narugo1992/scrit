import re
import time
from urllib.parse import urljoin, quote_plus

import requests

from ..utils import get_random_ua

SKEB_WEBISTE = 'https://skeb.jp'


class SkebRateLimitError(requests.HTTPError):
    """Skeb answered 429 and the ``request_key`` challenge did not help; the caller must back off."""


class SkebClient:
    """Thin client of the skeb.jp JSON API.

    Requests are issued one at a time and ``min_interval`` seconds apart. A 429 is answered at most
    twice with the ``request_key`` cookie challenge the site hands out; if that does not clear it,
    ``SkebRateLimitError`` is raised instead of retrying, so a caller can never hammer the site.
    """

    def __init__(self, min_interval: float = 0.0):
        self._session = requests.session()
        self._session.headers.update({
            'Referer': 'https://skeb.jp',
            'User-Agent': get_random_ua(),
            "Authorization": "Bearer null",
            "Accept": "application/json, text/plain, */*",
        })
        self.min_interval = min_interval
        self.request_count = 0
        self._last_request = 0.0

    def _pace(self):
        wait = self._last_request + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def _get(self, url, params=None):
        for attempt in range(3):
            self._pace()
            self.request_count += 1
            resp = self._session.get(urljoin(SKEB_WEBISTE, url), params=params or {})
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()

            if attempt < 2 and 'request_key' in resp.cookies:
                continue
            cookies = re.findall(r'document.cookie\s*=\s*"request_key=(?P<content>[^;]+);', resp.text)
            if attempt < 2 and cookies:
                self._session.cookies.update({'request_key': cookies[0]})
                continue
            break

        raise SkebRateLimitError(f'429 Too Many Requests for url: {resp.url}', response=resp)

    def get_page(self, offset: int = 0, limit: int = 90):
        return self._get(
            '/api/works',
            {
                'sort': 'date',
                'genre': 'art',
                'offset': offset,
                'limit': limit,
            }
        )

    def iter_art_pages(self, limit: int = 90):
        offset = 0
        while True:
            items = self.get_page(offset, limit)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_user_page(self, offset: int = 0, limit: int = 90, sort: str = 'popularity'):
        return self._get(
            '/api/users',
            {
                'sort': sort,
                'offset': offset,
                'limit': limit,
            }
        )

    def iter_user_pages(self, limit: int = 90, sort: str = 'popularity'):
        # sort : popularity / date / request_masters / first_requesters
        offset = 0
        while True:
            items = self.get_user_page(offset, limit, sort)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_user_info(self, screen_name: str):
        return self._get(f'/api/users/{quote_plus(screen_name)}')

    def get_work_page(self, screen_name: str, role: str = 'client', sort='date', offset: int = 0):
        return self._get(
            f'/api/users/{quote_plus(screen_name)}/works',
            {
                'role': role,
                'sort': sort,
                'offset': offset,
            }
        )

    def iter_work_pages(self, screen_name: str, role: str = 'client', sort='date'):
        # role : client/creator
        offset = 0
        while True:
            items = self.get_work_page(screen_name, role, sort, offset)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_post(self, username, post_id):
        return self._get(f'/api/users/{username}/works/{post_id}')
