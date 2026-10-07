import os
from types import SimpleNamespace

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.utils import HfHubHTTPError

os.environ.setdefault('REMOTE_REPOSITORY', 'user/repo')

from test.prepare import repack  # noqa: E402

GB = 1024 ** 3


class FakeClient:
    def __init__(self, failures=0):
        self.commits = []
        self.failures = failures

    def create_commit(self, repo_id, repo_type, operations, commit_message, commit_description=''):
        if self.failures:
            self.failures -= 1
            raise HfHubHTTPError('boom')
        self.commits.append((commit_message, operations))


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(repack, 'hf_client', client)
    monkeypatch.setattr(repack, '_ensure_repository', lambda: None)
    monkeypatch.setattr(repack, '_load_archived_ids', lambda: ['old_1'])
    monkeypatch.setattr(repack, '_make_records', lambda: [{'filename': 'pack_1.zip', 'size': 5}])
    monkeypatch.setattr(repack.time, 'sleep', lambda s: None)
    return client


def listing(monkeypatch, sizes):
    monkeypatch.setattr(repack, 'hf_repo_glob', lambda **kwargs: [
        SimpleNamespace(path=f'unarchived/{name}.zip', size=size) for name, size in sizes.items()])


@pytest.mark.unittest
class TestPromotion:
    def test_the_largest_oversized_zip_becomes_a_pack_by_copy(self, fake, monkeypatch):
        listing(monkeypatch, {'small': 1 * GB, 'big': 9 * GB, 'huge': 12 * GB, 'edge': 5.6 * GB})
        assert repack.promote_oversized() is True
        message, operations = fake.commits[0]
        copy = next(op for op in operations if isinstance(op, CommitOperationCopy))
        assert copy.src_path_in_repo == 'unarchived/huge.zip' and copy.path_in_repo.startswith('packs/pack_')
        assert [op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)] == ['unarchived/huge.zip']
        paths = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        assert paths == {'README.md', 'index.json', 'archived.json'}

    def test_nothing_to_do_without_oversized_zips(self, fake, monkeypatch):
        listing(monkeypatch, {'small': 1 * GB, 'other': 5 * GB})
        assert repack.promote_oversized() is False
        assert fake.commits == []

    def test_promotion_comes_first_in_a_round(self, fake, monkeypatch):
        listing(monkeypatch, {'big': 9 * GB})
        monkeypatch.setattr(repack, 'repack_zips', lambda **kwargs: pytest.fail('normal packing must wait'))
        assert repack.repack_all() is True

    def test_commit_is_retried_a_bounded_number_of_times(self, fake, monkeypatch):
        listing(monkeypatch, {'big': 9 * GB})
        fake.failures = repack.COMMIT_ATTEMPTS - 1
        assert repack.promote_oversized() is True
        fake.failures = repack.COMMIT_ATTEMPTS
        with pytest.raises(HfHubHTTPError):
            repack.promote_oversized()


@pytest.mark.unittest
def test_pack_commit_titles_say_what_went_in_and_how_big(fake, monkeypatch):
    listing(monkeypatch, {'big_one': 9 * GB})
    assert repack.promote_oversized() is True
    message = fake.commits[0][0]
    assert message.startswith('[pack] pack_') and '1 oversized res, 9.0 GiB, copied as is | big_one' in message
