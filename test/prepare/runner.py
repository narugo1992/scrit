import logging
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests
from hbutils.system import TemporaryDirectory

from pyskeb.client.client import SkebRateLimitError
from .errors import GenericException, NoContent, ResourceBlocked, ResourceGone, ResourceTransient
from .http import Fetcher
from .process import build_zip
from .sites import resolve
from .sites.common import host_of
from .store import CommitFailed, Store, LeaseUnavailable
from .url import extract_urls

# seconds until the next retry of a queued resource, indexed by the number of attempts made so far
RETRY_DELAYS = [3600, 6 * 3600, 24 * 3600, 3 * 24 * 3600, 7 * 24 * 3600, 14 * 24 * 3600]
SKEB_BAN_STEPS = [2 * 3600, 4 * 3600, 8 * 3600, 12 * 3600]
LIST_PAGE_SIZE = 90
LIST_MAX_OFFSET = 3900


def split_post_path(path: str) -> Tuple[str, int]:
    matching = re.fullmatch(r'/?@(?P<username>[\s\S]+?)/works/(?P<work_id>\d+?)/?', path)
    return matching.group('username'), int(matching.group('work_id'))


class LeaseLost(GenericException):
    """Another crawler took the lease over; this one stops at once."""


class SkebUnavailable(GenericException):
    """skeb.jp failed in a way that says nothing about the post itself (network error, 5xx)."""


@dataclass
class Job:
    url: str
    prefix: str
    rid: str
    site: object
    post: str
    attempts: int = 0
    first_seen: float = 0.0
    last_error: str = ''


@dataclass
class RunConfig:
    budget_seconds: float = 5.5 * 3600
    poll_interval: float = 600.0
    bootstrap: int = 400
    max_skeb_requests: int = 4500
    flush_every: int = 20
    flush_seconds: float = 120.0
    retry_batch: int = 20
    max_extra_posts: int = 100
    lease_every: float = 300.0
    once: bool = False


class Runner:
    def __init__(self, store: Store, skeb, fx: Fetcher, config: RunConfig,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 holder: Optional[str] = None, use_lease: bool = True):
        self.store = store
        self.holder = holder or f'{os.environ.get("GITHUB_RUN_ID", "local")}-{uuid.uuid4().hex[:8]}'
        self.use_lease = use_lease
        self._last_poll = 0.0
        self._last_lease = 0.0
        self.skeb = skeb
        self.fx = fx
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.cooldown: Dict[str, float] = {}
        self.deadline = clock() + config.budget_seconds
        self.counters: Dict[str, int] = {}
        self.discovered: List[str] = []
        self.bugs = 0
        self.stop_reason = ''
        self._posts_since_flush = 0

    # ---------------------------------------------------------------- helpers
    def _count(self, site: str, status: str):
        self.store.bump(site, status)
        self.counters[status] = self.counters.get(status, 0) + 1

    def _time_left(self) -> float:
        return self.deadline - self.clock()

    def _skeb_budget_left(self) -> bool:
        return self.skeb.request_count < self.config.max_skeb_requests

    # ---------------------------------------------------------------- one resource
    def _attempt(self, job: Job, workdir: str) -> Tuple[str, Optional[str]]:
        """Return ``(status, zip_path)``; status is one of ok / dup / gone / empty / deferred / retry / bug."""
        if self.store.known(job.rid):
            return 'dup', None
        until = self.cooldown.get(job.site.NAME, 0.0)
        if self.clock() < until:
            job.last_error = 'host cooling down'
            return 'deferred', None

        zip_path = os.path.join(workdir, f'{job.rid}.zip')
        try:
            build_zip(job.site, job.url, job.prefix, zip_path, self.fx)
        except NoContent as err:
            logging.info(f'{job.rid}: nothing to archive ({err})')
            job.last_error = str(err)
            return 'empty', None
        except ResourceGone as err:
            logging.info(f'{job.rid}: gone ({err})')
            job.last_error = str(err)
            return 'gone', None
        except ResourceBlocked as err:
            self.cooldown[job.site.NAME] = self.clock() + err.cooldown
            job.last_error = f'blocked: {err}'
            logging.warning(f'{job.site.NAME} blocked for {err.cooldown:.0f}s: {err}')
            return 'retry', None
        except (ResourceTransient, GenericException) as err:
            job.last_error = f'transient: {err}'
            logging.warning(f'{job.rid}: {err}')
            return 'retry', None
        except Exception as err:  # noqa: BLE001 - last line of defence at the resource boundary: a bug in one
            # handler must not stall the cursor of the whole crawl; it is logged loudly and fails the run at exit.
            job.last_error = f'bug: {err!r}'
            logging.error(f'{job.rid}: unexpected {err!r}\n{traceback.format_exc()}')
            print(f'::error title=handler bug::{job.rid}: {err!r}', flush=True)
            self.bugs += 1
            return 'bug', None
        return 'ok', zip_path

    def _queue(self, job: Job, status: str):
        """Put a failed job into the retry queue, or drop it when it ran out of attempts."""
        if status in ('retry', 'bug'):
            job.attempts += 1
        if job.attempts > len(RETRY_DELAYS):
            self._count(job.site.NAME, 'expired')
            self.store.drop_pending(job.rid)
            logging.info(f'{job.rid}: gave up after {job.attempts} attempts ({job.last_error})')
            return
        delay = RETRY_DELAYS[min(max(job.attempts - 1, 0), len(RETRY_DELAYS) - 1)]
        if status == 'deferred':
            delay = max(self.cooldown.get(job.site.NAME, 0.0) - self.clock(), 60.0)
        self.store.add_pending({
            'rid': job.rid, 'url': job.url, 'prefix': job.prefix, 'post': job.post, 'site': job.site.NAME,
            'attempts': job.attempts, 'first_seen': job.first_seen, 'last_error': job.last_error[:300],
            'next_try': self.clock() + delay,
        })

    def _handle_jobs(self, jobs: List[Job], message: str):
        """Run the jobs of one post (or one retry batch) and commit everything in a single commit."""
        zips: Dict[str, str] = {}
        done_jobs: List[Job] = []
        with TemporaryDirectory() as td:
            for job in jobs:
                status, zip_path = self._attempt(job, td)
                self._count(job.site.NAME, 'uploaded' if status == 'ok' else status)
                if status == 'ok':
                    zips[job.rid] = zip_path
                    done_jobs.append(job)
                elif status in ('retry', 'bug', 'deferred'):
                    self._queue(job, status)
                else:
                    self.store.drop_pending(job.rid)
                    self.store.done.add(job.rid)
                    if status in ('gone', 'empty'):
                        self.store.note_dropped(job.rid, status, job.last_error, job.post)
            try:
                self.store.commit(zips, message)
            except CommitFailed as err:
                logging.error(str(err))
                for job in done_jobs:
                    job.last_error = f'commit failed: {err}'
                    self._count(job.site.NAME, 'commit_failed')
                    self._queue(job, 'retry')
                try:
                    self.store.commit({}, message + ' (queue after failed commit)')
                except CommitFailed as state_err:
                    logging.error(f'state could not be saved either: {state_err}')
                return
            for job in done_jobs:
                self.store.drop_pending(job.rid)
                self.store.done.add(job.rid)
                logging.info(f'{job.rid}: uploaded')
        self._posts_since_flush = 0

    # ---------------------------------------------------------------- one post
    def process_post(self, path: str, from_listing: bool = True):
        username, work_id = split_post_path(path)
        try:
            post = self.skeb.get_post(username, work_id)
        except SkebRateLimitError:
            raise
        except (requests.ConnectionError, requests.Timeout) as err:
            raise SkebUnavailable(f'{path}: {err!r}') from err
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code >= 500:
                raise SkebUnavailable(f'{path}: {err!r}') from err
            logging.warning(f'{path}: cannot read the post ({err}), skipped')
            if from_listing:
                self.store.mark_done(path)
            return

        text = f"{post.get('source_body') or ''}\n{post.get('body') or ''}"
        jobs: List[Job] = []
        seen = set()
        for url in extract_urls(text):
            if from_listing and self._is_skeb_work(url):
                self._discover(url)
                continue
            resolved = resolve(url)
            if resolved is None:
                self.store.bump('unsupported', 'urls')
                self.store.note_unsupported(host_of(url))
                continue
            site, rid = resolved
            if rid in seen:
                continue
            seen.add(rid)
            jobs.append(Job(url=url, prefix=f'{username}_{work_id}_', rid=rid, site=site, post=path,
                            first_seen=self.clock()))
        logging.info(f'{path}: {len(jobs)} supported resource(s)')

        if from_listing:
            self.store.mark_done(path)
        if jobs:
            self._handle_jobs(jobs, f'newest: {path} +{len(jobs)} resource(s)')
        else:
            self._posts_since_flush += 1
            idle = time.time() - self.store.last_commit_at
            if self._posts_since_flush >= self.config.flush_every or idle >= self.config.flush_seconds:
                self.store.commit({}, 'newest: update state')
                self._posts_since_flush = 0

    # ---------------------------------------------------------------- linked works
    @staticmethod
    def _is_skeb_work(url: str) -> bool:
        return host_of(url) == 'skeb.jp' and re.fullmatch(r'/@[^/]+/works/\d+/?', re.sub(r'[?#].*$', '', url.split('skeb.jp', 1)[-1])) is not None

    def _discover(self, url: str):
        """A post that links another work usually quotes an earlier commission with its own references."""
        path = re.sub(r'[?#].*$', '', url.split('skeb.jp', 1)[-1]).rstrip('/')
        if path not in self.store.head and path not in self.discovered:
            self.discovered.append(path)

    def process_discovered(self):
        todo, self.discovered = self.discovered, []
        for path in todo[:self.config.max_extra_posts]:
            if self._time_left() < 180 or not self._skeb_budget_left():
                break
            if self.store.remember_extra(path):
                self.process_post(path, from_listing=False)
                self._count('skeb', 'linked_work')

    # ---------------------------------------------------------------- queue
    def process_pending(self):
        now = self.clock()
        due = [item for item in self.store.pending if item['next_try'] <= now][:self.config.retry_batch]
        for item in due:
            if self._time_left() < 120:
                break
            resolved = resolve(item['url'])
            if resolved is None:
                self.store.drop_pending(item['rid'])
                continue
            site, rid = resolved
            job = Job(url=item['url'], prefix=item['prefix'], rid=rid, site=site, post=item['post'],
                      attempts=item['attempts'], first_seen=item['first_seen'], last_error=item['last_error'])
            self._handle_jobs([job], f'newest: retry {rid} (attempt {job.attempts + 1})')

    # ---------------------------------------------------------------- listing
    def _page(self, offset: int) -> List[Dict]:
        try:
            return self.skeb.get_page(offset, LIST_PAGE_SIZE)
        except (requests.ConnectionError, requests.Timeout) as err:
            raise SkebUnavailable(f'listing at offset {offset}: {err!r}') from err
        except requests.HTTPError as err:
            if isinstance(err, SkebRateLimitError) or err.response is None or err.response.status_code < 500:
                raise
            raise SkebUnavailable(f'listing at offset {offset}: {err!r}') from err

    def collect_new_posts(self) -> List[str]:
        """Paths of posts newer than everything listed before, newest first."""
        head = self.store.known_posts()
        fresh: List[str] = []
        seen = set()
        known_streak = 0
        offset = 0
        while offset < LIST_MAX_OFFSET and self._skeb_budget_left():
            items = self._page(offset)
            if not items:
                break
            for item in items:
                path = item['path']
                if path in head:
                    known_streak += 1
                    if known_streak >= 3:
                        return fresh
                    continue
                known_streak = 0
                if path not in seen:
                    seen.add(path)
                    fresh.append(path)
            offset += len(items)
            if not head and len(fresh) >= self.config.bootstrap:
                return fresh[:self.config.bootstrap]
        return fresh

    # ---------------------------------------------------------------- skeb ban handling
    def _skeb_blocked(self) -> float:
        return float(self.store.state['skeb'].get('blocked_until') or 0.0)

    def _register_ban(self):
        streak = int(self.store.state['skeb'].get('ban_streak') or 0)
        wait = SKEB_BAN_STEPS[min(streak, len(SKEB_BAN_STEPS) - 1)]
        self.store.set_skeb(blocked_until=self.clock() + wait, ban_streak=streak + 1, last_ban=self.clock())
        logging.error(f'Skeb answered 429; no request to skeb.jp for the next {wait / 3600:.0f}h.')
        print(f'::warning title=skeb rate limited::paused skeb.jp requests for {wait / 3600:.0f}h', flush=True)

    # ---------------------------------------------------------------- main loop
    def _poll(self):
        """List the newest posts and put the new ones in front of the backlog."""
        fresh = self.collect_new_posts()
        self._last_poll = self.clock()
        if fresh:
            logging.info(f'{len(fresh)} new post(s) queued in front of {len(self.store.backlog)} waiting.')
            self.store.enqueue(fresh)

    def _keep_lease(self, force: bool = False):
        if not self.use_lease or (not force and self.clock() - self._last_lease < self.config.lease_every):
            return
        if not self.store.keep_lease(self.holder):
            raise LeaseLost('another crawler holds the lease now')
        self._last_lease = self.clock()

    def drain_backlog(self) -> int:
        """Process waiting posts newest first. A new poll happens every poll interval even in the middle of a long
        backlog, so posts that appear meanwhile are taken before the older ones that are still waiting."""
        processed = 0
        while self.store.backlog:
            if self._time_left() < 180 or not self._skeb_budget_left():
                self.stop_reason = self.stop_reason or 'budget'
                break
            self._keep_lease()
            if self.clock() - self._last_poll >= self.config.poll_interval:
                self._poll()
            self.process_post(self.store.backlog[0])
            processed += 1
        return processed

    def cycle(self) -> int:
        """One round: newest posts first, then the retry queue, then the older works that posts link to."""
        processed = 0
        skeb_ok = self.clock() >= self._skeb_blocked() and self._skeb_budget_left()
        failed = False
        pending_done = False
        if skeb_ok:
            try:
                self._poll()
                processed = self.drain_backlog()
                if not self.store.backlog:
                    self.process_pending()
                    pending_done = True
                    self.process_discovered()
                if self.store.state['skeb'].get('ban_streak'):
                    self.store.set_skeb(ban_streak=0)
            except SkebRateLimitError:
                self._register_ban()
                failed = True
            except SkebUnavailable as err:
                logging.warning(f'skeb.jp unavailable, the rest waits for the next poll: {err}')
                self.stop_reason = 'skeb unavailable'
                failed = True
        fresh_waiting = skeb_ok and not failed and bool(self.store.backlog)
        if not pending_done and not fresh_waiting:
            self.process_pending()  # nothing fresh can be fetched right now, the old failures may go ahead
        self.store.commit({}, 'newest: update state')
        return processed

    def run(self):
        self.store.refresh()
        self.store.load_state()
        if self.use_lease:
            self.store.acquire_lease(self.holder)
            self._last_lease = self.clock()
        try:
            while True:
                self.store.refresh()
                self._keep_lease()
                self.cycle()
                if self.config.once or self._time_left() < self.config.poll_interval + 180:
                    break
                self.stop_reason = ''
                self.sleep(self.config.poll_interval)
            self.store.commit({}, 'newest: update state')
        except LeaseLost as err:
            logging.error(str(err))
            self.stop_reason = 'lease lost'
            return
        finally:
            if self.use_lease and self.stop_reason != 'lease lost':
                self.store.release_lease(self.holder)

    def summary(self) -> str:
        lines = [f'newest run finished ({self.stop_reason or "time budget reached"}), '
                 f'skeb requests: {self.skeb.request_count}, bugs: {self.bugs}',
                 'this run: ' + ', '.join(f'{k}={v}' for k, v in sorted(self.counters.items())),
                 f'pending queue: {len(self.store.pending)}, posts still waiting: {len(self.store.backlog)}']
        return '\n'.join(lines)
