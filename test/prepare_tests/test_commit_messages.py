import re

import pytest

from test.prepare.errors import ResourceGone, ResourceTransient
from test.prepare.fmt import pretty_size, by_site, plural
from test.prepare_tests.test_runner import env, paths, MemStore  # noqa: F401  (fixtures and helpers)


@pytest.mark.unittest
class TestPrettySize:
    @pytest.mark.parametrize('size, text', [
        (0, '0 B'), (812, '812 B'), (1023, '1023 B'), (1024, '1.0 KiB'), (48_300_000, '46.1 MiB'),
        (5_905_562_159, '5.5 GiB'), (9.53e9, '8.9 GiB'), (2.847e12, '2.6 TiB'), (5e15, '4547.5 TiB'),
    ])
    def test_sizes(self, size, text):
        assert pretty_size(size) == text

    def test_helpers(self):
        assert plural(1, 'post') == '1 post' and plural(0, 'post') == '0 posts' and plural(3, 'post') == '3 posts'
        assert by_site(['imgur', 'google', 'google']) == 'google 2, imgur 1'


def titles(store):
    return [commit['message'] for commit in store.client.commits]


@pytest.mark.unittest
class TestCrawlerCommitMessages:
    def test_a_new_post_says_what_was_fetched_how_big_and_how_much_is_waiting(self, env):
        site, clock, make = env
        listing = paths(3)
        posts = {listing[0]: 'https://fake.test/a https://fake.test/b'}
        runner, store, skeb = make(listing, posts)
        store.state['backlog'] = []
        runner.process_post  # noqa: B018
        store.enqueue(listing)
        runner.process_post(listing[0])
        message = titles(store)[-1]
        assert re.fullmatch(r'\[new\] @u3/works/3 \| \+2 res, \d+ B \(fake 2\) \| 2 posts waiting', message), message
        description = store.client.commits[-1]['description'].splitlines()
        assert [line.split()[0] for line in description] == ['fake_a', 'fake_b'] and all(re.fullmatch(r'\S+  \d+ B', line) for line in description)

    def test_a_failure_is_reported_as_queued_not_as_fetched(self, env):
        site, clock, make = env
        listing = paths(1)
        site.behaviour = {'https://fake.test/slow': ResourceTransient('net'), 'https://fake.test/dead': ResourceGone('404')}
        runner, store, skeb = make(listing, {listing[0]: 'https://fake.test/slow https://fake.test/dead'})
        runner.cycle()
        assert titles(store)[0] == '[new] @u1/works/1 | +0 res, 1 gone or empty, 1 queued for retry | 0 posts waiting'
        assert store.client.commits[0]['description'] == ''

    def test_a_retry_is_labelled_with_its_attempt_and_the_post_it_came_from(self, env):
        site, clock, make = env
        runner, store, skeb = make(paths(1), {paths(1)[0]: 'no links'})
        store.add_pending({'rid': 'fake_old', 'url': 'https://fake.test/old', 'prefix': 'x_', 'post': '/@x/works/1',
                           'site': 'fake', 'attempts': 2, 'first_seen': 0, 'last_error': '', 'next_try': 0})
        runner.cycle()
        retry = [m for m in titles(store) if m.startswith('[retry]')]
        assert len(retry) == 1 and re.fullmatch(r'\[retry\] fake_old \(attempt 3, from @x/works/1\) \| \+1 res, \d+ B \(fake 1\)', retry[0]), retry

    def test_a_linked_older_work_is_marked_old(self, env):
        site, clock, make = env
        listing = ['/@a/works/10']
        posts = {'/@a/works/10': 'https://skeb.jp/@b/works/3', '/@b/works/3': 'https://fake.test/linked'}
        runner, store, skeb = make(listing, posts)
        runner.cycle()
        old = [m for m in titles(store) if m.startswith('[old]')]
        assert len(old) == 1 and old[0].startswith('[old] @b/works/3 (linked from a post) | +1 res'), old

    def test_posts_without_resources_are_summarised_in_a_state_commit(self, env):
        site, clock, make = env
        listing = paths(3)
        runner, store, skeb = make(listing, {p: 'nothing to see' for p in listing})
        runner.cycle()
        assert titles(store) == ['[state] 3 posts checked, nothing to fetch | 0 posts waiting']

    def test_lease_commits_are_readable(self):
        store = MemStore()
        store.acquire_lease('run-1', sleep=lambda s: None)
        store.release_lease('run-1')
        assert titles(store) == ['[lease] held by run local', '[lease] released']
