import os
import time
from typing import Dict, List

import pytest
import requests

from pyskeb.client.client import SkebRateLimitError
from test.prepare import runner as runner_module
from test.prepare.errors import ResourceBlocked, ResourceGone, ResourceTransient, NoContent
from test.prepare.runner import Runner, RunConfig, RETRY_DELAYS
from test.prepare.store import Store, CommitFailed


class FakeClient:
    def __init__(self):
        self.commits: List[Dict] = []
        self.fail_commits = 0
        self.files: Dict[str, bytes] = {}

    def create_commit(self, repo_id, repo_type, operations, commit_message, commit_description=''):
        if self.fail_commits:
            self.fail_commits -= 1
            raise requests.ConnectionError('boom')
        self.commits.append({
            'message': commit_message,
            'description': commit_description,
            'paths': [op.path_in_repo for op in operations],
        })
        for op in operations:
            if isinstance(op.path_or_fileobj, bytes):
                self.files[op.path_in_repo] = op.path_or_fileobj


class MemStore(Store):
    def __init__(self, archived=(), head=()):
        super().__init__(FakeClient(), 'user/repo')
        self.archived = set(archived)
        self.state = {'version': 1, 'head': list(head), 'backlog': [], 'stats': {}, 'skeb': {}}
        self._index_loaded_at = 1e18

    def _read_json(self, path, default):
        import json
        raw = self.client.files.get(path)
        return json.loads(raw) if raw is not None else default

    def load_state(self):
        pass


class FakeSkeb:
    def __init__(self, listing, posts):
        self.listing = listing
        self.posts = posts
        self.request_count = 0
        self.pages_asked = []
        self.ratelimited = False

    def get_page(self, offset, limit):
        self.request_count += 1
        self.pages_asked.append(offset)
        if self.ratelimited:
            raise SkebRateLimitError('429')
        return [{'path': path} for path in self.listing[offset:offset + limit]]

    def get_post(self, username, work_id):
        self.request_count += 1
        key = f'/@{username}/works/{work_id}'
        if self.ratelimited:
            raise SkebRateLimitError('429')
        if key not in self.posts:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError('404', response=response)
        return {'body': self.posts[key], 'source_body': ''}


class FakeSite:
    def __init__(self, name='fake', behaviour=None):
        self.NAME = name
        self.behaviour = behaviour or {}
        self.calls = []

    def match(self, url):
        return None

    def download(self, fx, url, out_dir):
        self.calls.append(url)
        outcome = self.behaviour.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        with open(os.path.join(out_dir, 'a.png'), 'wb') as f:
            f.write(b'data')


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def env(monkeypatch):
    site = FakeSite()
    monkeypatch.setattr(runner_module, 'resolve',
                        lambda url: (site, 'fake_' + url.rsplit('/', 1)[-1]) if 'fake.test' in url else None)
    clock = Clock()

    def make(listing, posts, store=None, **config):
        store = store or MemStore()
        skeb = FakeSkeb(listing, posts)
        cfg = RunConfig(budget_seconds=1000 * 24 * 3600, poll_interval=60, **config)
        runner = Runner(store, skeb, None, cfg, clock=clock, sleep=clock.sleep, use_lease=False)
        return runner, store, skeb

    return site, clock, make


def paths(n):
    return [f'/@u{i}/works/{i}' for i in range(n, 0, -1)]  # newest first


@pytest.mark.unittest
class TestRunner:
    def test_posts_are_processed_newest_first_and_cursor_covers_them(self, env):
        site, clock, make = env
        listing = paths(3)
        posts = {p: f'https://fake.test/r{i}' for i, p in enumerate(listing)}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        assert site.calls == ['https://fake.test/r0', 'https://fake.test/r1', 'https://fake.test/r2']
        assert store.head == listing and store.backlog == []  # newest first, nothing left waiting
        uploaded = [c for c in store.client.commits if any(p.startswith('unarchived/') for p in c['paths'])]
        assert len(uploaded) == 3
        # zips and state travel in the same commit
        assert all('state/newest.json' in c['paths'] for c in uploaded)

    def test_only_new_posts_after_the_cursor(self, env):
        site, clock, make = env
        listing = paths(6)
        store = MemStore(head=listing[3:])
        posts = {p: 'https://fake.test/x' + p[-1] for p in listing}
        runner, _, skeb = make(listing, posts, store=store)
        runner.cycle()
        assert site.calls == ['https://fake.test/x3', 'https://fake.test/x2', 'https://fake.test/x1'] or \
               len(site.calls) == 3
        assert skeb.pages_asked == [0]

    def test_bootstrap_limit_when_no_cursor(self, env):
        site, clock, make = env
        listing = paths(50)
        runner, store, skeb = make(listing, {}, bootstrap=5)
        runner.cycle()
        assert len(store.head) == 5

    def test_known_resource_is_not_downloaded(self, env):
        site, clock, make = env
        listing = paths(1)
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/seen'},
                                   store=MemStore(archived=['fake_seen']))
        runner.cycle()
        assert site.calls == []

    def test_transient_failure_goes_to_pending_and_is_retried(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/flaky': ResourceTransient('net')}
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/flaky'})
        runner.cycle()
        assert [item['rid'] for item in store.pending] == ['fake_flaky']
        assert store.pending[0]['attempts'] == 1
        assert store.pending[0]['next_try'] == pytest.approx(clock.now + RETRY_DELAYS[0], abs=5)

        runner.cycle()
        assert len(site.calls) == 1  # not due yet
        clock.now += RETRY_DELAYS[0] + 1
        site.behaviour = {}
        runner.cycle()
        assert store.pending == []
        assert 'fake_flaky' in store.done

    def test_gone_and_empty_are_dropped_without_retry(self, env):
        site, clock, make = env
        listing = paths(2)
        site.behaviour = {'https://fake.test/dead': ResourceGone('gone'), 'https://fake.test/blank': NoContent('none')}
        posts = {listing[1]: 'https://fake.test/dead', listing[0]: 'https://fake.test/blank'}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        assert store.pending == []

    def test_blocked_host_is_cooled_down_for_the_following_resources(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/a': ResourceBlocked('quota', cooldown=1500)}
        posts = {listing[0]: 'https://fake.test/a https://fake.test/b https://fake.test/c'}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        assert site.calls == ['https://fake.test/a']  # b and c were not even tried
        assert sorted(item['rid'] for item in store.pending) == ['fake_a', 'fake_b', 'fake_c']
        deferred = [item for item in store.pending if item['rid'] != 'fake_a']
        assert all(item['attempts'] == 0 for item in deferred)

    def test_attempts_expire(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/bad': ResourceTransient('net')}
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/bad'})
        runner.cycle()
        for _ in range(len(RETRY_DELAYS) + 1):
            clock.now += RETRY_DELAYS[-1] + 1
            runner.cycle()
        assert store.pending == []

    def test_skeb_429_pauses_requests_and_escalates(self, env):
        site, clock, make = env
        listing = paths(3)
        runner, store, skeb = make(listing, {})
        skeb.ratelimited = True
        runner.cycle()
        first = skeb.request_count
        assert first == 1
        assert store.state['skeb']['blocked_until'] > clock.now
        runner.cycle()
        assert skeb.request_count == first  # no request while paused
        clock.now = store.state['skeb']['blocked_until'] + 1
        runner.cycle()
        assert skeb.request_count == first + 1
        assert store.state['skeb']['ban_streak'] == 2

    def test_skeb_request_cap_stops_listing(self, env):
        site, clock, make = env
        listing = paths(300)
        runner, store, skeb = make(listing, {}, max_skeb_requests=2)
        runner.cycle()
        assert skeb.request_count <= 3

    def test_failed_commit_queues_the_resources(self, env, monkeypatch):
        site, clock, make = env
        monkeypatch.setattr('test.prepare.store.time.sleep', lambda s: None)
        listing = paths(1)
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/z'})
        store.client.fail_commits = 6
        runner.cycle()
        assert [item['rid'] for item in store.pending] == ['fake_z']
        assert 'fake_z' not in store.done

    def test_deleted_post_is_skipped_but_cursor_moves(self, env):
        site, clock, make = env
        listing = paths(2)
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/ok'})  # listing[1] -> 404
        runner.cycle()
        assert store.head == listing

    def test_handler_bug_is_counted_and_does_not_stall(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/bug': KeyError('x')}
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/bug'})
        runner.cycle()
        assert runner.bugs == 1
        assert store.head == listing
        assert store.pending[0]['attempts'] == 1


@pytest.mark.unittest
class TestDiscovery:
    def test_linked_work_is_processed_without_touching_the_cursor(self, env):
        site, clock, make = env
        listing = ['/@a/works/10']
        posts = {
            '/@a/works/10': 'see https://skeb.jp/@b/works/3?foo=1 and https://fake.test/main',
            '/@b/works/3': 'older request https://fake.test/old and https://skeb.jp/@c/works/9',
        }
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        assert sorted(site.calls) == ['https://fake.test/main', 'https://fake.test/old']
        assert store.head == ['/@a/works/10']  # the linked work never enters the cursor
        assert store.state['extra_seen'] == ['/@b/works/3']  # and its own skeb links are not chased

    def test_linked_work_is_handled_once(self, env):
        site, clock, make = env
        listing = ['/@a/works/10']
        posts = {'/@a/works/10': 'https://skeb.jp/@b/works/3', '/@b/works/3': 'https://fake.test/old'}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        store.state['head'] = []  # simulate a second sweep over the same listing
        runner.cycle()
        assert site.calls == ['https://fake.test/old']

    def test_unsupported_hosts_are_counted(self, env):
        site, clock, make = env
        listing = ['/@a/works/10']
        runner, store, skeb = make(listing, {'/@a/works/10': 'https://www.example.org/x https://example.org/y'})
        runner.cycle()
        assert store.state['unsupported_hosts'] == {'example.org': 2}


@pytest.mark.unittest
class TestFailureListIsCommittedWithEveryUpload:
    def test_failure_is_committed_in_the_same_pass_as_the_post(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/flaky': ResourceTransient('net')}
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/flaky'})
        runner.cycle()
        commits = store.client.commits
        assert commits and 'state/pending.json' in commits[0]['paths']  # not postponed to the end of the run
        assert not any(p.startswith('unarchived/') for p in commits[0]['paths'])

    def test_upload_commit_always_carries_state_and_pending(self, env):
        site, clock, make = env
        listing = paths(1)
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/good'})
        runner.cycle()
        first = store.client.commits[0]
        assert {'unarchived/fake_good.zip', 'state/newest.json', 'state/pending.json'} <= set(first['paths'])

    def test_cursor_is_flushed_after_a_quiet_period_even_without_resources(self, env):
        site, clock, make = env
        listing = paths(3)
        runner, store, skeb = make(listing, {listing[0]: 'no links here'}, flush_every=1000)
        store.last_commit_at = time.time() - 1000
        store.enqueue([listing[0]])
        runner.process_post(listing[0])
        assert any('state/newest.json' in c['paths'] for c in store.client.commits)


@pytest.mark.unittest
class TestSkebHiccups:
    def test_listing_network_error_does_not_crash_the_run(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(2), {})

        def broken(offset, limit):
            skeb.request_count += 1
            raise requests.ConnectionError('Remote end closed connection')

        skeb.get_page = broken
        runner.cycle()  # must not raise
        assert runner.stop_reason == 'skeb unavailable'

    def test_listing_5xx_waits_for_the_next_poll(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(2), {})

        def broken(offset, limit):
            response = requests.Response()
            response.status_code = 502
            raise requests.HTTPError('502', response=response)

        skeb.get_page = broken
        runner.cycle()
        assert runner.stop_reason == 'skeb unavailable'

    def test_client_retries_a_dropped_connection_once(self, monkeypatch):
        from pyskeb.client.client import SkebClient
        client = SkebClient()
        calls = []

        class Resp:
            status_code = 200
            cookies = {}

            def raise_for_status(self):
                pass

            def json(self):
                return {'ok': True}

        def flaky_get(url, params=None, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ConnectionError('closed')
            return Resp()

        monkeypatch.setattr(client._session, 'get', flaky_get)
        monkeypatch.setattr('pyskeb.client.client.time.sleep', lambda s: None)
        assert client._get('/api/works') == {'ok': True}
        assert len(calls) == 2

    def test_client_gives_up_after_the_second_failure(self, monkeypatch):
        from pyskeb.client.client import SkebClient
        client = SkebClient()
        monkeypatch.setattr(client._session, 'get', lambda url, params=None, timeout=None: (_ for _ in ()).throw(requests.ConnectionError('x')))
        monkeypatch.setattr('pyskeb.client.client.time.sleep', lambda s: None)
        with pytest.raises(requests.ConnectionError):
            client._get('/api/works')


@pytest.mark.unittest
class TestSkebTimeouts:
    def test_every_request_has_a_timeout(self, monkeypatch):
        from pyskeb.client.client import SkebClient
        client = SkebClient()
        seen = {}

        class Resp:
            status_code = 200
            cookies = {}

            def raise_for_status(self):
                pass

            def json(self):
                return []

        def get(url, params=None, timeout=None):
            seen['timeout'] = timeout
            return Resp()

        monkeypatch.setattr(client._session, 'get', get)
        client._get('/api/works')
        assert seen['timeout'] == (10, 60)

    def test_idle_connections_are_dropped_before_the_next_request(self, monkeypatch):
        from pyskeb.client.client import SkebClient
        client = SkebClient()
        closed = []

        class Resp:
            status_code = 200
            cookies = {}

            def raise_for_status(self):
                pass

            def json(self):
                return []

        monkeypatch.setattr(client._session, 'get', lambda url, params=None, timeout=None: Resp())
        monkeypatch.setattr(client._session, 'close', lambda: closed.append(1))
        client._last_request = time.time() - 600
        client._get('/api/works')
        assert closed == [1]
        client._get('/api/works')
        assert closed == [1]  # a request right after another one keeps the connection


@pytest.mark.unittest
class TestDroppedAudit:
    def test_given_up_resources_are_listed_with_their_reason(self, env):
        site, clock, make = env
        listing = paths(2)
        site.behaviour = {'https://fake.test/dead': ResourceGone('HTTP 404 for the thing'), 'https://fake.test/blank': NoContent('no media')}
        posts = {listing[1]: 'https://fake.test/dead', listing[0]: 'https://fake.test/blank'}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        dropped = {item['rid']: item for item in store.state['dropped']}
        assert dropped['fake_dead']['status'] == 'gone' and 'HTTP 404' in dropped['fake_dead']['why']
        assert dropped['fake_blank']['status'] == 'empty'
        assert dropped['fake_dead']['post'].startswith('/@')

    def test_audit_list_is_bounded(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(1), {})
        for index in range(500):
            store.note_dropped(f'r{index}', 'gone', 'x', '/@a/works/1', limit=300)
        assert len(store.state['dropped']) == 300 and store.state['dropped'][-1]['rid'] == 'r499'


@pytest.mark.unittest
class TestEmptyUnarchivedDirectory:
    def test_a_missing_directory_means_nothing_is_waiting(self):
        from huggingface_hub.utils import EntryNotFoundError

        class Client:
            def list_repo_tree(self, *args, **kwargs):
                raise EntryNotFoundError("Entry 'unarchived' not found in repository")

            def hf_hub_download(self, **kwargs):
                raise EntryNotFoundError('archived.json missing')

        store = Store(Client(), 'user/repo')
        store.refresh()  # must not raise
        assert store.unarchived == set() and store.archived == set()


@pytest.mark.unittest
class TestNewestHasPriority:
    def test_new_posts_overtake_the_backlog_that_is_still_waiting(self, env):
        site, clock, make = env
        old = paths(6)[3:]               # three older posts that are already waiting
        posts = {p: f'https://fake.test/{p[-1]}' for p in paths(6)}
        runner, store, skeb = make(paths(6), posts)
        store.state['head'] = list(old)
        store.state['backlog'] = list(old)
        runner.cycle()
        assert [c.rsplit('/', 1)[1] for c in site.calls] == ['6', '5', '4', '3', '2', '1']  # newest ... oldest, all of it

    def test_a_long_backlog_is_interrupted_by_polling_for_even_newer_posts(self, env):
        site, clock, make = env
        listing = paths(3)
        posts = {p: f'https://fake.test/{p[-1]}' for p in paths(5)}
        runner, store, skeb = make(listing, posts, )
        runner.config.poll_interval = 60
        original = runner.process_post
        arrivals = {'done': False}

        def slow_post(path, from_listing=True):
            clock.now += 100  # every post takes longer than a poll interval
            if not arrivals['done'] and path == '/@u3/works/3':
                skeb.listing = paths(5)  # two newer posts appear while the first one is being processed
                arrivals['done'] = True
            return original(path, from_listing)

        runner.process_post = slow_post
        runner.cycle()
        order = [c.rsplit('/', 1)[1] for c in site.calls]
        assert order == ['3', '5', '4', '2', '1']  # 5 and 4 jumped ahead of the older 2 and 1

    def test_an_interrupted_run_keeps_the_rest_and_the_next_run_still_goes_newest_first(self, env):
        site, clock, make = env
        listing = paths(4)
        posts = {p: f'https://fake.test/{p[-1]}' for p in paths(6)}
        runner, store, skeb = make(listing, posts)
        runner.config.budget_seconds = 10_000_000
        original = runner.process_post
        count = {'n': 0}

        def interrupted(path, from_listing=True):
            count['n'] += 1
            if count['n'] == 3:
                runner.deadline = clock.now  # the time budget ends after two posts
            return original(path, from_listing)

        runner.process_post = interrupted
        runner.cycle()
        assert [c.rsplit('/', 1)[1] for c in site.calls] == ['4', '3', '2']
        assert store.backlog == ['/@u1/works/1']  # what is left is the oldest post

        second, store2, skeb2 = make(paths(6), posts, store=store)
        second.process_post = original.__func__.__get__(second)
        site.calls.clear()
        second.cycle()
        assert [c.rsplit('/', 1)[1] for c in site.calls][:2] == ['6', '5']  # newest arrivals come before the leftovers

    def test_retries_wait_until_every_fresh_post_is_done(self, env):
        site, clock, make = env
        listing = paths(2)
        posts = {listing[0]: 'https://fake.test/fresh0', listing[1]: 'https://fake.test/fresh1'}
        runner, store, skeb = make(listing, posts)
        store.add_pending({'rid': 'fake_old', 'url': 'https://fake.test/old', 'prefix': 'x_', 'post': '/@x/works/1',
                           'site': 'fake', 'attempts': 1, 'first_seen': 0, 'last_error': '', 'next_try': 0})
        runner.cycle()
        assert site.calls == ['https://fake.test/fresh0', 'https://fake.test/fresh1', 'https://fake.test/old']

    def test_retries_do_not_run_while_fresh_posts_are_still_waiting(self, env):
        site, clock, make = env
        listing = paths(3)
        posts = {p: f'https://fake.test/f{p[-1]}' for p in listing}
        runner, store, skeb = make(listing, posts)
        store.add_pending({'rid': 'fake_old', 'url': 'https://fake.test/old', 'prefix': 'x_', 'post': '/@x/works/1',
                           'site': 'fake', 'attempts': 1, 'first_seen': 0, 'last_error': '', 'next_try': 0})
        runner.deadline = clock.now + 100  # not enough time left to finish: only two posts fit
        original = runner.process_post

        def one_post_then_out_of_time(path, from_listing=True):
            result = original(path, from_listing)
            runner.deadline = clock.now  # budget gone after the first post
            return result

        runner.process_post = one_post_then_out_of_time
        runner.cycle()
        assert 'https://fake.test/old' not in site.calls and store.backlog  # the old failure waited for the fresh ones

    def test_linked_old_works_come_after_the_retry_queue(self, env):
        site, clock, make = env
        listing = ['/@a/works/10']
        posts = {'/@a/works/10': 'https://skeb.jp/@b/works/3', '/@b/works/3': 'https://fake.test/linked'}
        runner, store, skeb = make(listing, posts)
        store.add_pending({'rid': 'fake_old', 'url': 'https://fake.test/old', 'prefix': 'x_', 'post': '/@x/works/1',
                           'site': 'fake', 'attempts': 1, 'first_seen': 0, 'last_error': '', 'next_try': 0})
        runner.cycle()
        assert site.calls == ['https://fake.test/old', 'https://fake.test/linked']

    def test_when_skeb_is_unavailable_the_retry_queue_still_goes_ahead(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(2), {})
        store.add_pending({'rid': 'fake_old', 'url': 'https://fake.test/old', 'prefix': 'x_', 'post': '/@x/works/1',
                           'site': 'fake', 'attempts': 1, 'first_seen': 0, 'last_error': '', 'next_try': 0})

        def down(offset, limit):
            skeb.request_count += 1
            raise requests.ConnectionError('down')

        skeb.get_page = down
        runner.cycle()
        assert site.calls == ['https://fake.test/old']


@pytest.mark.unittest
class TestLease:
    def make_store(self, client=None):
        store = MemStore()
        if client is not None:
            store.client = client
        return store

    def lease_of(self, store):
        return store.read_lease()

    def test_a_free_lease_is_taken_and_released(self):
        store = self.make_store()
        store.acquire_lease('run-1', sleep=lambda s: None)
        assert self.lease_of(store)['holder'] == 'run-1'
        store.release_lease('run-1')
        assert self.lease_of(store)['holder'] is None

    def test_a_live_holder_keeps_a_second_crawler_out(self):
        from test.prepare.store import LeaseUnavailable
        client = FakeClient()
        first, second = self.make_store(client), self.make_store(client)
        first.acquire_lease('run-1', sleep=lambda s: None)
        waited = []
        with pytest.raises(LeaseUnavailable):
            second.acquire_lease('run-2', wait_limit=0.0, sleep=waited.append)
        assert self.lease_of(first)['holder'] == 'run-1'

    def test_a_dead_holder_is_replaced_after_the_ttl(self):
        import json
        from test.prepare import store as store_module
        client = FakeClient()
        client.files['state/lease.json'] = json.dumps({'holder': 'dead', 'at': time.time() - store_module.LEASE_TTL - 5}).encode()
        store = self.make_store(client)
        store.acquire_lease('run-2', sleep=lambda s: None)
        assert self.lease_of(store)['holder'] == 'run-2'

    def test_the_loser_of_a_simultaneous_write_waits(self):
        client = FakeClient()
        store = self.make_store(client)
        import json

        def rival_writes_during_the_settle_time(seconds):
            if seconds < 5:  # the settle sleep, not the wait between attempts
                client.files['state/lease.json'] = json.dumps({'holder': 'rival', 'at': time.time()}).encode()

        from test.prepare.store import LeaseUnavailable
        with pytest.raises(LeaseUnavailable):
            store.acquire_lease('run-1', wait_limit=0.0, sleep=rival_writes_during_the_settle_time, settle=1.0)

    def test_keeping_the_lease_fails_when_someone_else_took_it(self):
        import json
        client = FakeClient()
        store = self.make_store(client)
        store.acquire_lease('run-1', sleep=lambda s: None)
        assert store.keep_lease('run-1') is True
        client.files['state/lease.json'] = json.dumps({'holder': 'usurper', 'at': time.time()}).encode()
        assert store.keep_lease('run-1') is False
        store.release_lease('run-1')  # must not clear somebody else's lease
        assert self.lease_of(store)['holder'] == 'usurper'

    def test_a_run_stops_when_it_loses_the_lease(self, env):
        import json
        site, clock, make = env
        runner, store, skeb = make(paths(3), {p: 'https://fake.test/x' + p[-1] for p in paths(3)})
        runner.use_lease = True
        runner.config.lease_every = 0.0
        runner.sleep = lambda s: None
        store.acquire_lease = lambda holder, **kw: None
        original_keep = store.keep_lease
        store.keep_lease = lambda holder: False  # somebody else took over
        runner.run()
        assert runner.stop_reason == 'lease lost' and site.calls == []


@pytest.mark.unittest
class TestLeaseOfAFinishedRun:
    def test_a_lease_of_a_cancelled_run_is_taken_over_at_once(self):
        import json
        client = FakeClient()
        client.files['state/lease.json'] = json.dumps({'holder': 'old-1', 'at': time.time(), 'run': '12345'}).encode()
        store = MemStore()
        store.client = client
        store.holder_status = lambda run_id: 'completed' if run_id == '12345' else None
        waits = []
        store.acquire_lease('new-2', sleep=waits.append)
        assert store.read_lease()['holder'] == 'new-2' and 30.0 not in waits  # no waiting between attempts

    def test_a_lease_of_a_running_run_is_respected(self):
        import json
        from test.prepare.store import LeaseUnavailable
        client = FakeClient()
        client.files['state/lease.json'] = json.dumps({'holder': 'old-1', 'at': time.time(), 'run': '12345'}).encode()
        store = MemStore()
        store.client = client
        store.holder_status = lambda run_id: 'in_progress'
        with pytest.raises(LeaseUnavailable):
            store.acquire_lease('new-2', wait_limit=0.0, sleep=lambda s: None)

    def test_a_holder_whose_run_is_still_going_is_only_waited_for_briefly(self):
        import json
        from test.prepare.store import LeaseUnavailable
        client = FakeClient()
        client.files['state/lease.json'] = json.dumps({'holder': 'old-1', 'at': time.time(), 'run': '12345'}).encode()
        store = MemStore()
        store.client = client
        store.holder_status = lambda run_id: 'in_progress'
        started = time.time()
        with pytest.raises(LeaseUnavailable):
            store.acquire_lease('new-2', alive_wait=0.0, sleep=lambda s: None)  # the long wait limit must not apply
        assert time.time() - started < 5

    def test_an_unknown_holder_is_waited_for_until_it_expires(self):
        import json
        from test.prepare.store import LeaseUnavailable
        client = FakeClient()
        client.files['state/lease.json'] = json.dumps({'holder': 'laptop', 'at': time.time(), 'run': ''}).encode()
        store = MemStore()
        store.client = client
        waits = []
        with pytest.raises(LeaseUnavailable):
            store.acquire_lease('new-2', wait_limit=0.5, alive_wait=0.0, sleep=lambda s: (waits.append(s), time.sleep(0.3)))
        assert waits.count(30.0) >= 1  # it did wait, because nothing says that laptop is gone

    def test_release_is_one_write_and_only_for_the_holder(self):
        client = FakeClient()
        store = MemStore()
        store.client = client
        store.acquire_lease('run-1', sleep=lambda s: None)
        before = len(client.commits)
        store.release_lease('run-1')
        assert len(client.commits) == before + 1  # no read-then-write round trips that a kill could cut short
        store.release_lease('run-1')
        assert len(client.commits) == before + 1  # and not twice

    def test_the_github_status_check_is_conservative(self, monkeypatch):
        import requests as rq
        from test.prepare import ci
        monkeypatch.delenv('GITHUB_REPOSITORY', raising=False)
        assert ci.github_run_status('123') is None  # not on GitHub: unknown
        monkeypatch.setenv('GITHUB_REPOSITORY', 'a/b')
        monkeypatch.setenv('GITHUB_TOKEN', 'x')

        class Resp:
            def __init__(self, status, body):
                self.status_code, self._body = status, body

            def json(self):
                return self._body

        monkeypatch.setattr(ci.requests, 'get', lambda *a, **k: Resp(200, {'status': 'completed'}))
        assert ci.github_run_status('123') == 'completed'
        monkeypatch.setattr(ci.requests, 'get', lambda *a, **k: Resp(200, {'status': 'in_progress'}))
        assert ci.github_run_status('123') == 'in_progress'
        monkeypatch.setattr(ci.requests, 'get', lambda *a, **k: Resp(404, {}))
        assert ci.github_run_status('123') is None

        def boom(*a, **k):
            raise rq.ConnectionError('x')

        monkeypatch.setattr(ci.requests, 'get', boom)
        assert ci.github_run_status('123') is None
        assert ci.github_run_status('local') is None  # not a numeric run id


@pytest.mark.unittest
class TestEndingSafely:
    def test_the_hard_deadline_ends_the_run_with_code_5(self):
        with pytest.raises(SystemExit) as info:
            Runner._on_hard_deadline(None, None)
        assert info.value.code == 5

    def test_arming_sets_an_alarm_after_the_budget_and_a_forced_exit_after_that(self, env, monkeypatch):
        site, clock, make = env
        runner, store, skeb = make(paths(1), {})
        runner.config.budget_seconds, runner.config.hard_extra = 1000, 600
        seen = {}
        monkeypatch.setattr('test.prepare.runner.signal.signal', lambda sig, handler: seen.setdefault('handler', handler))
        monkeypatch.setattr('test.prepare.runner.signal.alarm', lambda n: seen.setdefault('alarm', n))

        class FakeTimer:
            def __init__(self, interval, function, args):
                seen['force_after'], seen['force_with'] = interval, args

            daemon = False

            def start(self):
                seen['started'] = True

            def cancel(self):
                seen['cancelled'] = True

        monkeypatch.setattr('test.prepare.runner.threading.Timer', FakeTimer)
        runner._arm_hard_deadline()
        assert seen['alarm'] == 1600 and seen['force_after'] == 1690 and seen['force_with'] == (5,) and seen['started']
        runner._disarm_hard_deadline()
        assert seen['cancelled']

    def test_a_cancelled_run_releases_the_lease_on_the_way_out(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(1), {})
        runner.use_lease = True
        runner.sleep = lambda s: None
        store.acquire_lease = lambda holder, **kw: setattr(store, '_holding', holder)
        released = []
        store.release_lease = lambda holder: released.append(holder)
        store.keep_lease = lambda holder: True

        def killed(*a, **k):
            raise SystemExit(143)  # what SIGTERM turns into

        runner.cycle = killed
        with pytest.raises(SystemExit):
            runner.run()
        assert released == [runner.holder]

    def test_a_hub_outage_while_saving_state_does_not_end_the_crawler(self, env, monkeypatch):
        site, clock, make = env
        monkeypatch.setattr('test.prepare.store.time.sleep', lambda s: None)
        runner, store, skeb = make(paths(2), {p: 'no links' for p in paths(2)})
        store.client.fail_commits = 1000
        runner.cycle()  # must not raise
        assert store.dirty  # nothing was lost, the state is still waiting to be written
        store.client.fail_commits = 0
        runner._flush()
        assert not store.dirty

    def test_a_resource_that_takes_too_long_is_given_up_for_a_retry(self, tmp_path, monkeypatch):
        import socket
        import requests
        from test.prepare.errors import ResourceTransient
        from test.prepare.http import Fetcher
        monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port: [(2, 1, 6, '', ('93.184.216.34', 0))])
        fx = Fetcher(gap=0)

        class Slow(requests.Response):
            def iter_content(self, size):
                yield b'x' * 10
                yield b'x' * 10

        def fake_request(method, url, allow_redirects, **kwargs):
            import io
            resp = Slow()
            resp.raw = io.BytesIO(b'')
            resp.status_code = 200
            resp.headers['Content-Type'] = 'image/png'
            resp.url = url
            return resp

        monkeypatch.setattr(fx.session, 'request', fake_request)
        fx.begin_resource(seconds=1)
        fx.resource_deadline = time.time() - 1  # the time budget has run out
        with pytest.raises(ResourceTransient):
            fx.download('https://example.com/a.png', str(tmp_path / 'a'))
        assert not (tmp_path / 'a').exists()
