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
