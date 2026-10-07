import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set

import requests
from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from .errors import GenericException

STATE_PATH = 'state/newest.json'
PENDING_PATH = 'state/pending.json'
HEAD_SIZE = 600
PENDING_LIMIT = 3000
COMMIT_ATTEMPTS = 6


class CommitFailed(GenericException):
    pass


def _now_text() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


class Store:
    """The dataset repository seen by ``newest``.

    ``unarchived/`` receives the resource zips and ``state/`` keeps the crawler bookkeeping (cursor,
    counters, retry queue). Both are written in the same commit so they never disagree. ``packs/``,
    ``archived.json``, ``index.json`` and ``README.md`` belong to the repacker and are only read here.
    """

    def __init__(self, client, repo_id: str):
        self.client = client
        self.repo_id = repo_id
        self.archived: Set[str] = set()
        self.unarchived: Set[str] = set()
        self.done: Set[str] = set()
        self.state: Dict = {}
        self.pending: List[Dict] = []
        self._dirty_state = False
        self._dirty_pending = False
        self._index_loaded_at = 0.0
        self.last_commit_at = time.time()

    # ------------------------------------------------------------ reading
    def _read_json(self, path: str, default):
        try:
            local = self.client.hf_hub_download(
                repo_id=self.repo_id, repo_type='dataset', filename=path, force_download=True)
        except EntryNotFoundError:
            return default
        with open(local, 'r') as f:
            return json.load(f)

    def refresh(self, index_ttl: float = 3600.0):
        """Reload dedupe indexes (at most once per ``index_ttl`` seconds) and the state files once."""
        if time.time() - self._index_loaded_at >= index_ttl:
            self.archived = set(self._read_json('archived.json', []))
            self.unarchived = {
                os.path.splitext(os.path.basename(item.path))[0]
                for item in self._list_unarchived()
                if item.path.endswith('.zip')
            }
            self._index_loaded_at = time.time()
            logging.info(f'Dedupe index loaded: {len(self.archived)} archived, {len(self.unarchived)} unarchived.')

    def _list_unarchived(self):
        try:
            return list(self.client.list_repo_tree(self.repo_id, path_in_repo='unarchived', repo_type='dataset'))
        except EntryNotFoundError:
            # git keeps no empty directories: unarchived/ vanishes while the repacker has emptied it
            return []

    def load_state(self):
        self.state = self._read_json(STATE_PATH, {})
        self.state.setdefault('version', 1)
        self.state.setdefault('head', [])
        self.state.setdefault('stats', {})
        self.state.setdefault('skeb', {})
        self.pending = (self._read_json(PENDING_PATH, {}) or {}).get('items', [])

    def known(self, resource_id: str) -> bool:
        return resource_id in self.archived or resource_id in self.unarchived or resource_id in self.done

    # ------------------------------------------------------------ bookkeeping
    @property
    def head(self) -> List[str]:
        return self.state['head']

    def push_head(self, path: str):
        head = self.state['head']
        if path in head:
            head.remove(path)
        head.insert(0, path)
        del head[HEAD_SIZE:]
        self._dirty_state = True

    def bump(self, site: str, status: str, amount: int = 1):
        stats = self.state['stats']
        for key in ('total', site):
            bucket = stats.setdefault(key, {})
            bucket[status] = bucket.get(status, 0) + amount
        self._dirty_state = True

    def note_dropped(self, resource_id: str, status: str, reason: str, post: str, limit: int = 300):
        """Keep the latest resources that were given up on, so a wrong 'gone' can be audited afterwards."""
        dropped = self.state.setdefault('dropped', [])
        dropped.append({'rid': resource_id, 'status': status, 'why': reason[:160], 'post': post,
                        'at': _now_text()})
        del dropped[:-limit]
        self._dirty_state = True

    def note_unsupported(self, host: str):
        """Count links to hosts without a handler, to see which site is worth adding next."""
        hosts = self.state.setdefault('unsupported_hosts', {})
        hosts[host] = hosts.get(host, 0) + 1
        if len(hosts) > 400:
            for name, _ in sorted(hosts.items(), key=lambda item: item[1])[:100]:
                del hosts[name]
        self._dirty_state = True

    def remember_extra(self, path: str, limit: int = 4000) -> bool:
        """Remember a work found through a link; returns False when it was already handled."""
        seen = self.state.setdefault('extra_seen', [])
        if path in seen:
            return False
        seen.append(path)
        del seen[:-limit]
        self._dirty_state = True
        return True

    def set_skeb(self, **values):
        self.state['skeb'].update(values)
        self._dirty_state = True

    def add_pending(self, item: Dict):
        self.pending = [old for old in self.pending if old['rid'] != item['rid']]
        self.pending.append(item)
        del self.pending[:-PENDING_LIMIT]
        self._dirty_pending = True

    def drop_pending(self, resource_id: str):
        before = len(self.pending)
        self.pending = [old for old in self.pending if old['rid'] != resource_id]
        self._dirty_pending |= len(self.pending) != before

    @property
    def dirty(self) -> bool:
        return self._dirty_state or self._dirty_pending

    # ------------------------------------------------------------ writing
    def commit(self, zips: Optional[Dict[str, str]] = None, message: str = 'newest: update state'):
        """One atomic commit with the new zips and the state files that changed."""
        zips = zips or {}
        if not zips and not self.dirty:
            return
        operations = [CommitOperationAdd(path_in_repo=f'unarchived/{rid}.zip', path_or_fileobj=path)
                      for rid, path in zips.items()]
        if self._dirty_state or zips:
            self.state['updated_at'] = _now_text()
            operations.append(CommitOperationAdd(
                path_in_repo=STATE_PATH,
                path_or_fileobj=json.dumps(self.state, ensure_ascii=False, indent=1).encode('utf-8')))
        if self._dirty_pending or zips:
            operations.append(CommitOperationAdd(
                path_in_repo=PENDING_PATH,
                path_or_fileobj=json.dumps({'version': 1, 'items': self.pending}, ensure_ascii=False,
                                           indent=1).encode('utf-8')))

        for attempt in range(1, COMMIT_ATTEMPTS + 1):
            try:
                self.client.create_commit(repo_id=self.repo_id, repo_type='dataset', operations=operations,
                                          commit_message=message)
            except (HfHubHTTPError, requests.ConnectionError, requests.Timeout) as err:
                if attempt == COMMIT_ATTEMPTS:
                    raise CommitFailed(f'commit {message!r} failed {attempt} times: {err!r}') from err
                delay = min(20 * 2 ** (attempt - 1), 300)
                logging.warning(f'Commit failed ({err!r}), retry in {delay}s ({attempt}/{COMMIT_ATTEMPTS}) ...')
                time.sleep(delay)
            else:
                break

        self.unarchived.update(zips)
        self._dirty_state = self._dirty_pending = False
        self.last_commit_at = time.time()
