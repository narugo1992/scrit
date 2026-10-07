import pytest

from test.prepare.errors import ResourceBlocked, ResourceGone, ResourceTransient
from test.prepare.sites import google

FOLDER = 'application/vnd.google-apps.folder'


class Resp:
    def __init__(self, status=200, url='https://drive.google.com/drive/folders/x', text='<html></html>'):
        self.status_code = status
        self.url = url
        self.text = text


class Fx:
    def __init__(self, response):
        self.response = response

    def get(self, url, **kwargs):
        return self.response


class Node:
    def __init__(self, name):
        self.name = name


@pytest.mark.unittest
class TestFolderPage:
    def test_unparsable_page_is_gone_not_a_bug(self, monkeypatch):
        def broken(url, content):
            raise RuntimeError('Cannot retrieve the folder information from the link.')

        monkeypatch.setattr(google, '_parse_google_drive_file', broken)
        with pytest.raises(ResourceGone):
            google._parse_page(Fx(Resp()), 'abc')

    def test_login_redirect_means_private(self):
        with pytest.raises(ResourceGone):
            google._parse_page(Fx(Resp(url='https://accounts.google.com/ServiceLogin?x=1')), 'abc')

    def test_not_found(self):
        with pytest.raises(ResourceGone):
            google._parse_page(Fx(Resp(status=404)), 'abc')

    def test_throttling_is_a_cooldown(self):
        with pytest.raises(ResourceBlocked):
            google._parse_page(Fx(Resp(status=429)), 'abc')
        with pytest.raises(ResourceBlocked):
            google._parse_page(Fx(Resp(url='https://www.google.com/sorry/index?continue=x')), 'abc')

    def test_server_error_is_transient(self):
        with pytest.raises(ResourceTransient):
            google._parse_page(Fx(Resp(status=500)), 'abc')


@pytest.mark.unittest
class TestFolderListing:
    def _patch(self, monkeypatch, tree):
        monkeypatch.setattr(google, '_parse_page', lambda fx, fid: tree[fid])

    def test_nested_folders_keep_the_root_name_like_before(self, monkeypatch):
        self._patch(monkeypatch, {
            'root': ('Refs', [('f1', 'a.png', 'image/png'), ('sub', 'Sub', FOLDER)]),
            'sub': ('Sub', [('f2', 'b', 'image/jpeg')]),
        })
        entries = google._list_folder(None, 'root')
        assert [(item[1], item[2]) for item in entries] == [
            (['Refs', 'a.png'], 'image/png'),
            (['Refs', 'Sub', 'b.jpg'], 'image/jpeg'),
        ]

    def test_full_page_switches_to_the_drive_api(self, monkeypatch):
        children = [(f'f{i}', f'{i}.png', 'image/png') for i in range(50)]
        self._patch(monkeypatch, {'root': ('Big', children)})
        monkeypatch.setattr(google, '_list_folder_api', lambda fx, fid, name: [(f'f{i}', [name, f'{i}.png'], 'image/png') for i in range(130)])
        assert len(google._list_folder(None, 'root')) == 130

    def test_small_folders_never_touch_the_api(self, monkeypatch):
        self._patch(monkeypatch, {'root': ('Small', [('f1', 'a.png', 'image/png')])})
        monkeypatch.setattr(google, '_list_folder_api', lambda *a: pytest.fail('API must not be used'))
        assert len(google._list_folder(None, 'root')) == 1

    def test_shorter_api_result_keeps_the_page_listing(self, monkeypatch):
        children = [(f'f{i}', f'{i}.png', 'image/png') for i in range(50)]
        self._patch(monkeypatch, {'root': ('Big', children)})
        monkeypatch.setattr(google, '_list_folder_api', lambda fx, fid, name: [])
        assert len(google._list_folder(None, 'root')) == 50


@pytest.mark.unittest
class TestIdsAreAscii:
    def test_japanese_text_glued_to_a_folder_link_is_not_part_of_the_id(self):
        url = 'https://drive.google.com/drive/folders/1qr-WAGRFTPpRG06V8xVN6NNUmxIrqyx5の闘麗装'
        assert google.match(url) == 'googledrive_1qr-WAGRFTPpRG06V8xVN6NNUmxIrqyx5'


@pytest.mark.unittest
class TestDocumentExports:
    def test_drawings_are_recognised_and_exported_as_png(self, monkeypatch):
        url = 'https://docs.google.com/drawings/d/1KKv_e7sGB2ztkg5JpU0Xe_Tk9BN99_RC9RPKGouNjSc/edit?usp=drive_link'
        assert google.match(url) == 'googledrive_1KKv_e7sGB2ztkg5JpU0Xe_Tk9BN99_RC9RPKGouNjSc'
        fetched = []
        monkeypatch.setattr(google, 'fetch_file', lambda fx, u, out_dir, name=None, **kw: fetched.append((u, name)))
        google.download(None, url, '/tmp/x')
        assert fetched == [('https://docs.google.com/drawings/d/1KKv_e7sGB2ztkg5JpU0Xe_Tk9BN99_RC9RPKGouNjSc/export/png',
                            '1KKv_e7sGB2ztkg5JpU0Xe_Tk9BN99_RC9RPKGouNjSc.png')]

    def test_documents_keep_the_format_query(self, monkeypatch):
        fetched = []
        monkeypatch.setattr(google, 'fetch_file', lambda fx, u, out_dir, name=None, **kw: fetched.append(u))
        google.download(None, 'https://docs.google.com/document/d/1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa/edit', '/tmp/x')
        assert fetched == ['https://docs.google.com/document/d/1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa/export?format=docx']
