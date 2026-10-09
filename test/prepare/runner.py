import logging
import os
import re
import shutil
import signal
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests
from hbutils.system import TemporaryDirectory

from pyskeb.client.client import SkebRateLimitError
from .fmt import pretty_size, plural, by_site
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
class Wave:
    """Results collected since the last commit: the zips already on disk and the bookkeeping that goes with them.

    Nothing here is visible to the dataset or to the saved state until the wave is committed in one go.
    """
    dir: Optional[str] = None
    zips: Dict[str, str] = field(default_factory=dict)          # resource id -> zip path
    jobs: List[Tuple['Job', str, str]] = field(default_factory=list)  # (job, kind, label) of the uploaded ones
    bytes: int = 0
    dup: int = 0
    gone: int = 0
    queued: int = 0
    opened_at: Optional[float] = None


@dataclass
class RunConfig:
    budget_seconds: float = 5.5 * 3600
    poll_interval: float = 600.0
    bootstrap: int = 400
    max_skeb_requests: int = 4500
    wave_seconds: float = 180.0     # a wave is committed at the latest this long after its first result
    wave_resources: int = 60        # ... or as soon as it holds this many resources
    wave_bytes: float = 1024 ** 3   # ... or this many bytes of zips
    retry_batch: int = 20
    max_extra_posts: int = 100
    lease_every: float = 300.0
    hard_extra: float = 600.0       # seconds after the budget at which the run is ended whatever it is doing
    hard_deadline: bool = False
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
        self._checked = 0
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
        self._wave = Wave()

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
        if job.rid in self._wave.zips:
            return 'staged', None  # the same resource linked from another post, already in the open wave
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

    def _wave_dir(self) -> str:
        if self._wave.dir is None:
            self._wave.dir = tempfile.mkdtemp(prefix='wave_')
        return self._wave.dir

    def _handle_jobs(self, jobs: List[Job], kind: str, label: str):
        """Run the jobs of one post (or one retry) and add what came out of them to the open wave."""
        wave = self._wave
        workdir = self._wave_dir()
        for job in jobs:
            self._keep_lease()
            status, zip_path = self._attempt(job, workdir)
            if status == 'staged':
                continue
            self._count(job.site.NAME, 'uploaded' if status == 'ok' else status)
            if status == 'ok':
                wave.zips[job.rid] = zip_path
                wave.bytes += os.path.getsize(zip_path)
                wave.jobs.append((job, kind, label))
            elif status == 'dup':
                wave.dup += 1
                self.store.drop_pending(job.rid)
                self.store.done.add(job.rid)
            elif status in ('retry', 'bug', 'deferred'):
                self._queue(job, status)
                wave.queued += 1
            else:
                self.store.drop_pending(job.rid)
                self.store.done.add(job.rid)
                if status in ('gone', 'empty'):
                    self.store.note_dropped(job.rid, status, job.last_error, job.post)
                    wave.gone += 1
        self._maybe_close_wave()

    def _wave_title(self) -> str:
        """``[wave] +12 res, 48.3 MiB (googledrive 9, imgur 3) | new 9, retry 2, old 1 | 37 posts waiting``"""
        wave = self._wave
        notes = [text for text in (f'{wave.dup} already archived' if wave.dup else '',
                                   f'{wave.gone} gone or empty' if wave.gone else '',
                                   f'{wave.queued} queued for retry' if wave.queued else '') if text]
        if wave.zips:
            total = sum(os.path.getsize(path) for path in wave.zips.values())
            kinds = ', '.join(f'{name} {count}' for name in ('new', 'retry', 'old')
                              if (count := sum(1 for _, kind, _ in wave.jobs if kind == name)))
            head = (f'[wave] +{len(wave.zips)} res, {pretty_size(total)} '
                    f'({by_site(job.site.NAME for job, _, _ in wave.jobs)}) | {kinds}')
        else:
            head = f'[state] {plural(self._checked, "post")} checked, nothing to fetch' if self._checked \
                else '[state] queue and cursor updated'
        parts = [head] + ([', '.join(notes)] if notes else []) + [f'{plural(len(self.store.backlog), "post")} waiting']
        return ' | '.join(parts)

    def _reset_wave(self):
        for path in self._wave.zips.values():
            if os.path.exists(path):
                os.remove(path)
        self._wave = Wave(dir=self._wave.dir)
        self._checked = 0

    def _discard_wave(self):
        """Forget the open wave without committing it (the run is ending without the lease, or is over)."""
        if self._wave.dir is not None:
            shutil.rmtree(self._wave.dir, ignore_errors=True)
        self._wave = Wave()

    def _maybe_close_wave(self):
        """Commit the open wave once it is full or three minutes old. A round also closes it, see ``cycle``."""
        wave = self._wave
        now = self.clock()
        if wave.opened_at is None and (wave.zips or self.store.dirty):
            wave.opened_at = now
        if wave.opened_at is None:
            return
        full = len(wave.zips) >= self.config.wave_resources or wave.bytes >= self.config.wave_bytes
        if full or now - wave.opened_at >= self.config.wave_seconds:
            self._close_wave()

    def _drop_archived_meanwhile(self):
        """Before the commit: reload the dedupe indexes and keep out the staged zips that are already in the dataset.

        The indexes are read once an hour, so a resource archived by a repack in the meantime (a retry of a commit
        that had in fact gone through, say) could be staged again. Such a zip is dropped here, not uploaded twice.
        """
        wave = self._wave
        if not wave.zips:
            return
        self.store.refresh(index_ttl=0)
        for job, kind, label in list(wave.jobs):
            if job.rid not in self.store.archived and job.rid not in self.store.unarchived:
                continue
            path = wave.zips.pop(job.rid)
            if os.path.exists(path):
                os.remove(path)
            wave.jobs.remove((job, kind, label))
            wave.dup += 1
            self.store.drop_pending(job.rid)
            self.store.done.add(job.rid)
            self._count(job.site.NAME, 'dup')
            logging.warning(f'{job.rid}: already in the dataset, not uploaded again')

    def _close_wave(self):
        """Commit the open wave: its zips and the state in ONE commit, then settle the jobs that went in."""
        wave = self._wave
        if not wave.zips and not self.store.dirty:
            self._reset_wave()
            return
        self._drop_archived_meanwhile()
        description = '\n'.join(f'{job.rid}  {pretty_size(os.path.getsize(wave.zips[job.rid]))}  {kind}  {label}'
                                for job, kind, label in wave.jobs)
        message = self._wave_title()
        if not wave.zips:
            try:
                self.store.commit({}, message)
            except CommitFailed as err:
                # the hub is having trouble; the state stays dirty and goes out with the next wave
                logging.error(f'state not saved for now: {err}')
            self._reset_wave()
            return
        try:
            self.store.commit(wave.zips, message, description)
        except CommitFailed as err:
            # the zips did not go up, so none of their jobs is done: they all go back to the retry queue
            logging.error(str(err))
            for job, _, _ in wave.jobs:
                job.last_error = f'commit failed: {err}'
                self._count(job.site.NAME, 'commit_failed')
                self._queue(job, 'retry')
            failed = len(wave.jobs)
            self._reset_wave()
            try:
                self.store.commit({}, f'[wave] upload failed, {plural(failed, "resource")} queued for retry')
            except CommitFailed as state_err:
                logging.error(f'state could not be saved either: {state_err}')
            return
        for job, _, _ in wave.jobs:
            self.store.drop_pending(job.rid)
            self.store.done.add(job.rid)
            logging.info(f'{job.rid}: uploaded')
        self._reset_wave()

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
        label = path.lstrip('/') if from_listing else f'{path.lstrip("/")} (linked from a post)'
        if jobs:
            self._handle_jobs(jobs, 'new' if from_listing else 'old', label)
        else:
            self._checked += 1
            self._maybe_close_wave()

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
            if item['rid'] in self._wave.zips:
                continue  # already in the open wave, it is committed with it
            resolved = resolve(item['url'])
            if resolved is None:
                self.store.drop_pending(item['rid'])
                continue
            site, rid = resolved
            job = Job(url=item['url'], prefix=item['prefix'], rid=rid, site=site, post=item['post'],
                      attempts=item['attempts'], first_seen=item['first_seen'], last_error=item['last_error'])
            self._handle_jobs([job], 'retry', f'{rid} (attempt {job.attempts + 1}, from {job.post.lstrip("/")})')

    def seed_from_history(self) -> int:
        """Queue the failures recorded before the rework once, so they are retried with the current fetchers.

        They go in with a clean attempt count and are due at once; the queue is only worked when no fresh post
        is waiting, so they never delay new posts. Returns the number of resources queued.
        """
        if self.store.state.get('history_seeded'):
            return 0
        queued = 0
        for item in self.store.read_failed_history():
            resolved = resolve(item['url'])
            if resolved is None:
                continue
            site, rid = resolved
            if self.store.known(rid) or any(old['rid'] == rid for old in self.store.pending):
                continue
            username, work_id = item['post']
            self.store.add_pending({
                'rid': rid, 'url': item['url'], 'prefix': item['prefix'], 'post': f'/@{username}/works/{work_id}',
                'site': site.NAME, 'attempts': 0, 'first_seen': self.clock(),
                'last_error': ','.join(item.get('kinds') or [])[:300], 'next_try': 0.0,
            })
            queued += 1
        self.store.state['history_seeded'] = True
        self.store._dirty_state = True
        logging.info(f'Queued {queued} failures from before the rework for a retry.')
        return queued

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

    @staticmethod
    def _on_hard_deadline(signum, frame):
        print('::error title=hard deadline::the run is over its time limit and ends now so the next one can start',
              flush=True)
        raise SystemExit(5)

    def _arm_hard_deadline(self):
        """GitHub kills a job after 6 hours and skips whatever should run afterwards, so the crawler ends itself well
        before: first by a signal (clean exit, lease released), and if that cannot get through, by force."""
        seconds = int(self.config.budget_seconds + self.config.hard_extra)
        signal.signal(signal.SIGALRM, self._on_hard_deadline)
        signal.alarm(seconds)
        self._force_exit = threading.Timer(seconds + 90, os._exit, args=(5,))
        self._force_exit.daemon = True
        self._force_exit.start()

    def _disarm_hard_deadline(self):
        signal.alarm(0)
        timer = getattr(self, '_force_exit', None)
        if timer is not None:
            timer.cancel()

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
        self._close_wave()  # a round ends with a commit, so nothing waits through the sleep until the next round
        return processed

    def run(self):
        self.store.refresh()
        self.store.load_state()
        if self.use_lease:
            self.store.acquire_lease(self.holder)
            self._last_lease = self.clock()
        self.seed_from_history()
        if self.config.hard_deadline:
            self._arm_hard_deadline()
        try:
            while True:
                self.store.refresh()
                self._keep_lease()
                self.cycle()
                if self.config.once or self._time_left() < self.config.poll_interval + 180:
                    break
                self.stop_reason = ''
                self.sleep(self.config.poll_interval)
            self._close_wave()
        except LeaseLost as err:
            logging.error(str(err))
            self.stop_reason = 'lease lost'
            self._discard_wave()
            return
        finally:
            self._discard_wave()
            if self.config.hard_deadline:
                self._disarm_hard_deadline()
            if self.use_lease and self.stop_reason != 'lease lost':
                self.store.release_lease(self.holder)

    def summary(self) -> str:
        lines = [f'newest run finished ({self.stop_reason or "time budget reached"}), '
                 f'skeb requests: {self.skeb.request_count}, bugs: {self.bugs}',
                 'this run: ' + ', '.join(f'{k}={v}' for k, v in sorted(self.counters.items())),
                 f'pending queue: {len(self.store.pending)}, posts still waiting: {len(self.store.backlog)}']
        return '\n'.join(lines)
