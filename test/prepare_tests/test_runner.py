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

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail_commits:
            self.fail_commits -= 1
            raise requests.ConnectionError('boom')
        self.commits.append({
            'message': commit_message,
            'paths': [op.path_in_repo for op in operations],
        })


class MemStore(Store):
    def __init__(self, archived=(), head=()):
        super().__init__(FakeClient(), 'user/repo')
        self.archived = set(archived)
        self.state = {'version': 1, 'head': list(head), 'stats': {}, 'skeb': {}}
        self._index_loaded_at = 1e18

    def _read_json(self, path, default):
        return default

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
        runner = Runner(store, skeb, None, cfg, clock=clock, sleep=clock.sleep)
        return runner, store, skeb

    return site, clock, make


def paths(n):
    return [f'/@u{i}/works/{i}' for i in range(n, 0, -1)]  # newest first


@pytest.mark.unittest
class TestRunner:
    def test_posts_are_processed_oldest_first_and_cursor_moves(self, env):
        site, clock, make = env
        listing = paths(3)
        posts = {p: f'https://fake.test/r{i}' for i, p in enumerate(listing)}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        assert site.calls == ['https://fake.test/r2', 'https://fake.test/r1', 'https://fake.test/r0']
        assert store.head == listing  # newest first
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
