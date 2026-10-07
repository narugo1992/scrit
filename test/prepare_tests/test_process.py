import io
import os
import zipfile

import pytest

from test.prepare.errors import NoContent
from test.prepare.process import write_zip, build_zip


class FakeSite:
    NAME = 'fake'

    def __init__(self, files):
        self.files = files

    def download(self, fx, url, out_dir):
        for relpath, content in self.files.items():
            path = os.path.join(out_dir, relpath)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)


@pytest.mark.unittest
class TestZip:
    def test_naming_follows_the_existing_layout(self, tmp_path):
        site = FakeSite({
            'Folder A/sub dir/ref sheet (1).png': b'a',
            'top-level.JPG': b'b',
            '三面図.png': b'c',
        })
        zip_file = str(tmp_path / 'x.zip')
        build_zip(site, 'https://example.com', 'user_12_', zip_file, None)
        with zipfile.ZipFile(zip_file) as zf:
            assert sorted(zf.namelist()) == sorted([
                'user_12_Folder_A_sub_dir_ref_sheet_1.png',
                'user_12_top_level.JPG',
                'user_12_三面図.png',
            ])

    def test_collisions_are_kept_apart(self, tmp_path):
        site = FakeSite({'a b.png': b'1', 'a_b.png': b'2', 'a-b.png': b'3'})
        zip_file = str(tmp_path / 'x.zip')
        build_zip(site, 'u', 'p_', zip_file, None)
        with zipfile.ZipFile(zip_file) as zf:
            assert len(set(zf.namelist())) == 3

    def test_symbol_only_name_gets_a_fallback(self, tmp_path):
        site = FakeSite({'★★.png': b'1'})
        build_zip(site, 'u', 'p_', str(tmp_path / 'x.zip'), None)
        with zipfile.ZipFile(str(tmp_path / 'x.zip')) as zf:
            assert zf.namelist() == ['p_file.png']

    def test_empty_result_raises_and_leaves_nothing(self, tmp_path):
        zip_file = str(tmp_path / 'x.zip')
        with pytest.raises(NoContent):
            build_zip(FakeSite({}), 'u', 'p_', zip_file, None)
        assert not os.path.exists(zip_file)

    def test_partial_download_files_are_ignored(self, tmp_path):
        site = FakeSite({'.part_abc': b'x', 'ok.png': b'y'})
        build_zip(site, 'u', 'p_', str(tmp_path / 'x.zip'), None)
        with zipfile.ZipFile(str(tmp_path / 'x.zip')) as zf:
            assert zf.namelist() == ['p_ok.png']


@pytest.mark.unittest
class TestHeaders:
    def test_latin1_mojibake_is_undone(self):
        from test.prepare.http import filename_from_headers
        raw = '三面図.png'.encode('utf-8').decode('latin-1')
        assert filename_from_headers({'Content-Disposition': f'attachment; filename="{raw}"'}) == '三面図.png'

    def test_proper_unicode_and_ascii_are_kept(self):
        from test.prepare.http import filename_from_headers
        assert filename_from_headers({'Content-Disposition': "attachment; filename*=UTF-8''%E4%B8%89%E9%9D%A2%E5%9B%B3.png"}) == '三面図.png'
        assert filename_from_headers({'Content-Disposition': 'attachment; filename="plain.png"'}) == 'plain.png'
        assert filename_from_headers({}) is None


@pytest.mark.unittest
class TestFetcherSafety:
    def _fx(self, monkeypatch, address):
        import socket
        from test.prepare.http import Fetcher
        monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port: [(2, 1, 6, '', (address, 0))])
        return Fetcher()

    @pytest.mark.parametrize('url', [
        'http://169.254.169.254/latest/meta-data', 'http://[::1]/x', 'ftp://example.com/x', 'file:///etc/passwd',
    ])
    def test_unsafe_urls_are_refused(self, monkeypatch, url):
        from test.prepare.errors import UnsafeUrl
        with pytest.raises(UnsafeUrl):
            self._fx(monkeypatch, '93.184.216.34').check_public(url)

    @pytest.mark.parametrize('address', ['10.0.0.5', '127.0.0.1', '169.254.169.254', '192.168.1.1', '100.64.0.1'])
    def test_hosts_resolving_to_private_addresses_are_refused(self, monkeypatch, address):
        from test.prepare.errors import UnsafeUrl
        with pytest.raises(UnsafeUrl):
            self._fx(monkeypatch, address).check_public('https://evil.example.com/x')

    def test_public_hosts_pass(self, monkeypatch):
        self._fx(monkeypatch, '93.184.216.34').check_public('https://example.com/x')

    def test_redirect_to_private_host_is_refused(self, monkeypatch):
        import requests
        from test.prepare.errors import UnsafeUrl
        from test.prepare.http import Fetcher

        fx = Fetcher(gap=0)
        answers = {'good.example.com': '93.184.216.34', 'bad.example.com': '10.1.1.1'}
        import socket
        monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port: [(2, 1, 6, '', (answers[host], 0))])

        def fake_request(method, url, allow_redirects, **kwargs):
            resp = requests.Response()
            resp.raw = io.BytesIO(b'')
            resp.status_code = 302
            resp.headers['Location'] = 'http://bad.example.com/secret'
            resp.url = url
            return resp

        monkeypatch.setattr(fx.session, 'request', fake_request)
        with pytest.raises(UnsafeUrl):
            fx.get('https://good.example.com/start')

    def test_resource_budget(self, tmp_path, monkeypatch):
        import requests
        from test.prepare.errors import TooLarge
        from test.prepare.http import Fetcher
        import socket
        monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port: [(2, 1, 6, '', ('93.184.216.34', 0))])
        fx = Fetcher(gap=0)

        class Raw(requests.Response):
            def iter_content(self, size):
                yield b'x' * 600

        def fake_request(method, url, allow_redirects, **kwargs):
            resp = Raw()
            resp.raw = io.BytesIO(b'')
            resp.status_code = 200
            resp.headers['Content-Type'] = 'image/png'
            resp.url = url
            return resp

        monkeypatch.setattr(fx.session, 'request', fake_request)
        fx.begin_resource(limit=1000)
        fx.download('https://example.com/a.png', str(tmp_path / 'a'))
        with pytest.raises(TooLarge):
            fx.download('https://example.com/b.png', str(tmp_path / 'b'))
        assert not (tmp_path / 'b').exists()


@pytest.mark.unittest
class TestDropboxUnzip:
    def test_oversized_archives_are_kept_packed(self, tmp_path, monkeypatch):
        import zipfile
        from test.prepare.sites import dropbox
        monkeypatch.setattr(dropbox, '_MAX_MEMBERS', 2)
        monkeypatch.setattr(dropbox, 'fetch_file', lambda fx, url, out_dir: _make_zip(out_dir, 5))
        dropbox.download(None, 'https://www.dropbox.com/scl/fo/abc/def?rlkey=x&dl=0', str(tmp_path))
        assert os.listdir(str(tmp_path)) == ['pack.zip']

    def test_small_archives_are_unpacked(self, tmp_path, monkeypatch):
        from test.prepare.sites import dropbox
        monkeypatch.setattr(dropbox, 'fetch_file', lambda fx, url, out_dir: _make_zip(out_dir, 2))
        dropbox.download(None, 'https://www.dropbox.com/scl/fo/abc/def?rlkey=x&dl=0', str(tmp_path))
        assert sorted(os.listdir(str(tmp_path))) == ['f0.txt', 'f1.txt']


def _make_zip(out_dir, count):
    path = os.path.join(out_dir, 'pack.zip')
    with zipfile.ZipFile(path, 'w') as zf:
        for index in range(count):
            zf.writestr(f'f{index}.txt', 'x')
    return path
