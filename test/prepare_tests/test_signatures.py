import ast
import glob
import inspect

import pytest

from test.prepare.sites import hosts


@pytest.mark.unittest
def test_every_fetch_all_call_matches_its_signature():
    """The hosts handlers talk to live sites, so a wrong call would only show up in production."""
    expected = len(inspect.signature(hosts._fetch_all).parameters)
    checked = 0
    for path in glob.glob(hosts.__file__):
        for node in ast.walk(ast.parse(open(path).read())):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_fetch_all':
                assert len(node.args) + len(node.keywords) == expected, f'{path}:{node.lineno}'
                checked += 1
    assert checked >= 10


@pytest.mark.unittest
def test_every_site_exposes_the_handler_interface():
    from test.prepare.sites import SITES
    for site in SITES:
        assert isinstance(site.NAME, str) and site.NAME
        assert callable(site.match) and callable(site.download)
        assert len(inspect.signature(site.download).parameters) == 3, site.NAME


@pytest.mark.unittest
class TestRefererAndPlaceholders:
    def test_hotlink_protected_hosts_get_their_own_referer(self):
        from test.prepare.http import Fetcher
        assert Fetcher._with_referer('https://i.ibb.co/x/y.png', {})['headers']['Referer'] == 'https://ibb.co/'
        assert Fetcher._with_referer('https://postimg.cc/abc/token?dl=1', {'headers': {'X': '1'}})['headers'] == \
            {'X': '1', 'Referer': 'https://postimg.cc/'}

    def test_an_explicit_referer_wins_and_other_hosts_are_untouched(self):
        from test.prepare.http import Fetcher
        explicit = {'headers': {'referer': 'https://elsewhere/'}}
        assert Fetcher._with_referer('https://i.ibb.co/x.png', explicit) is explicit
        assert Fetcher._with_referer('https://example.com/x.png', {}) == {}

    def test_a_180px_answer_from_a_prone_host_is_rejected_and_removed(self, tmp_path):
        from PIL import Image
        from test.prepare.errors import ResourceTransient
        path = str(tmp_path / 'thumb.png')
        Image.new('RGB', (180, 129)).save(path)
        with pytest.raises(ResourceTransient):
            hosts._check_not_a_placeholder('https://i.postimg.cc/x/y.png', path)
        assert not (tmp_path / 'thumb.png').exists()

    def test_real_images_and_other_hosts_pass(self, tmp_path):
        from PIL import Image
        big, small = str(tmp_path / 'big.png'), str(tmp_path / 'small.png')
        Image.new('RGB', (1200, 800)).save(big)
        Image.new('RGB', (64, 64)).save(small)
        hosts._check_not_a_placeholder('https://i.ibb.co/x/big.png', big)
        hosts._check_not_a_placeholder('https://files.catbox.moe/small.png', small)  # tiny but not from a prone host
        (tmp_path / 'z.zip').write_bytes(b'PK\x03\x04 not an image')
        hosts._check_not_a_placeholder('https://i.ibb.co/x/z.zip', str(tmp_path / 'z.zip'))  # unreadable header: kept
