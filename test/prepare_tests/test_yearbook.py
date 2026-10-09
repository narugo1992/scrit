import datetime as dt
import io
import json
import os
import zipfile

import pytest

from test.prepare import yearbook as yb


def make_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buffer.getvalue()


def range_of(blob):
    def get(start, end):
        return blob[start:end + 1]
    return get


@pytest.mark.unittest
class TestCentralDirectory:
    def test_members_are_listed_with_sizes_from_the_end_of_the_zip_only(self):
        blob = make_zip([('a_1_one.png', b'x' * 100), ('a_1_two.psd', b'y' * 2000), ('sub/dir/c.mp4', b'z' * 5)])
        reads = []

        def get(start, end):
            reads.append((start, end))
            return blob[start:end + 1]
        members = yb.read_members(len(blob), get)
        assert sorted(members) == sorted([('a_1_one.png', 100), ('a_1_two.psd', 2000), ('sub/dir/c.mp4', 5)])
        assert all(end - start < len(blob) for start, end in reads)  # never the whole file in one go

    def test_directory_entries_are_not_counted(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('folder/', b'')
            zf.writestr('folder/file.jpg', b'123')
        assert yb.read_members(len(buffer.getvalue()), range_of(buffer.getvalue())) == [('folder/file.jpg', 3)]

    def test_a_file_that_is_not_a_zip_is_refused(self):
        with pytest.raises(ValueError):
            yb.read_members(40, range_of(b'0' * 40))


@pytest.mark.unittest
class TestClassification:
    @pytest.mark.parametrize('name, category', [
        ('a_1_x.PNG', 'image'), ('a_1_x.jpeg', 'image'), ('a_1_x.gif', 'animated'), ('a_1_x.psd', 'layered'),
        ('a_1_x.clip', 'layered'), ('a_1_x.mp4', 'video'), ('a_1_x.mp3', 'audio'), ('a_1_x.pdf', 'document'),
        ('a_1_x.vrm', 'model'), ('a_1_x.zip', 'archive'), ('a_1_x.xyz', 'other'), ('README', 'other'),
    ])
    def test_extensions_map_to_big_categories(self, name, category):
        assert yb.category_of(name) == category


@pytest.mark.unittest
class TestTimes:
    def test_pack_names_are_utc_and_shown_in_utc_plus_eight(self):
        stamp = yb.pack_time('pack_20240501_125415_430239.zip')
        assert stamp.isoformat() == '2024-05-01T20:54:15+08:00'

    def test_unknown_names_have_no_time(self):
        assert yb.pack_time('something.zip') is None

    def test_quarter_and_month_buckets_follow_the_local_time(self):
        records = [
            yb.summarize_pack('pack_20240331_170000_000001.zip', 10, [('a_1_x.png', 5)]),  # 2024-04-01 01:00 local
            yb.summarize_pack('pack_20240401_000000_000001.zip', 10, [('a_1_y.psd', 7)]),  # 2024-04-01 08:00 local
        ]
        stats = yb.aggregate(records)
        assert set(stats['quarters']) == {'2024Q2'} and stats['quarters']['2024Q2']['files'] == 2
        assert stats['days']['2024-04-01']['files'] == 2
        assert stats['totals']['categories']['layered'] == 1

    def test_quarter_prefix_for_monthly_rows(self):
        assert yb.quarter_months_prefix('2024Q3') == '2024-07'


@pytest.mark.unittest
class TestPages:
    def test_file_names_link_to_the_file_page_not_the_download(self):
        stats = yb.aggregate([yb.summarize_pack('pack_20240501_125415_430239.zip', 1, [('a_1_x.png', 3)])])
        record = yb.summarize_pack('pack_20240501_125415_430239.zip', 1, [('a_1_x.png', 3)])
        page = yb.render_quarter('2024Q2', {'2024-05': [record]}, stats)
        assert '[p20240501125415]: https://hub.deepghs.org/datasets/hk1901/fuck_the_skeb/blob/main/packs/pack_20240501_125415_430239.zip' in page
        assert '/resolve/' not in page
        assert '[pack_20240501_125415_430239.zip][p20240501125415]' in page
        assert page.count('| [') == 1  # one row per zip, not per file
        assert '| [pack_20240501_125415_430239.zip][p20240501125415] (05-01 20:54) | 1 B | 3 B | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |' in page

    def test_pipes_and_brackets_in_file_names_are_escaped(self):
        assert yb._cell('a|b[c]') == 'a\\|b\\[c\\]'

    def test_readme_points_to_quarter_pages_and_charts(self, tmp_path):
        pack = 'pack_20240501_125415_430239.zip'
        records = [yb.summarize_pack(pack, 1, [('a_1_x.png', 3), ('a_1_y.psd', 9)])]
        written = yb.render_pages(records, str(tmp_path), ['2024Q2'])
        readme = (tmp_path / 'README.md').read_text(encoding='utf-8')
        assert 'index/2024Q2.md' in readme and './stats/quarters.png' in readme
        assert {'README.md', 'index/2024Q2.md', 'stats/quarters.png', 'stats/months_12.png',
                'stats/days_30.png', 'stats/categories.png', yb.MANIFEST_PATH} <= set(written)
        assert json.loads((tmp_path / yb.MANIFEST_PATH).read_text(encoding='utf-8'))[0]['files'] == 2

    def test_only_the_asked_quarters_are_rendered(self, tmp_path):
        records = [yb.summarize_pack('pack_20240501_125415_430239.zip', 1, [('a.png', 1)]),
                   yb.summarize_pack('pack_20250101_125415_430239.zip', 1, [('b.png', 1)])]
        written = yb.render_pages(records, str(tmp_path), ['2025Q1'])
        assert 'index/2025Q1.md' in written and 'index/2024Q2.md' not in written

    def test_quarter_tables_carry_the_size_of_each_category(self, tmp_path):
        records = [yb.summarize_pack('pack_20240501_125415_430239.zip', 1, [('a.png', 300), ('b.psd', 700)])]
        stats = yb.aggregate(records)
        page = yb.render_quarter('2024Q2', {}, stats)
        assert '| layered | 1 | 50.0% | 700 B | 70.0% |' in page
        assert '| image | 1 | 50.0% | 300 B | 30.0% |' in page


@pytest.mark.unittest
class TestManifest:
    def test_a_new_pack_changes_only_its_quarter_and_the_overview(self):
        known = [yb.summarize_pack('pack_20240501_125415_430239.zip', 1, [('a_1_x.png', 3)])]
        new = yb.summarize_pack('pack_20250101_125415_430239.zip', 1, [('b_2_y.psd', 7)])
        files = yb.plan_update(known, [new])
        assert set(files) == {'README.md', 'index/2025Q1.md', 'stats/quarters.png', 'stats/months_12.png',
                              'stats/days_30.png', 'stats/categories.png', yb.MANIFEST_PATH}
        assert 'index/2024Q2.md' not in files
        manifest = json.loads(files[yb.MANIFEST_PATH].decode('utf-8'))
        assert [r['name'] for r in manifest] == ['pack_20240501_125415_430239.zip', 'pack_20250101_125415_430239.zip']

    def test_members_of_a_local_zip_are_listed_without_directories(self, tmp_path):
        path = tmp_path / 'pack.zip'
        path.write_bytes(make_zip([('a/', b''), ('a/one.png', b'123'), ('two.mp4', b'12345')]))
        assert sorted(yb.zip_members_local(str(path))) == [('a/one.png', 3), ('two.mp4', 5)]


@pytest.mark.unittest
class TestOrder:
    def test_packs_and_months_run_newest_first(self):
        older = yb.summarize_pack('pack_20240501_100000_000001.zip', 1, [('a.png', 1)])
        newer = yb.summarize_pack('pack_20240520_100000_000001.zip', 1, [('b.png', 1)])
        earlier_month = yb.summarize_pack('pack_20240401_100000_000001.zip', 1, [('c.png', 1)])
        stats = yb.aggregate([older, newer, earlier_month])
        page = yb.render_quarter('2024Q2', {'2024-05': [older, newer], '2024-04': [earlier_month]}, stats)
        assert page.index('pack_20240520') < page.index('pack_20240501')  # newer pack above the older one
        assert page.index('## 2024-05') < page.index('## 2024-04')  # newer month above the older one

    def test_monthly_and_daily_tables_run_newest_first(self):
        records = [yb.summarize_pack(f'pack_2024{m:02d}01_100000_000001.zip', 1, [('a.png', 1)]) for m in (1, 3, 2)]
        stats = yb.aggregate(records)
        readme = yb.render_readme(stats, ['2024Q1'], {'categories': 'c.png', 'quarters': 'q.png', 'months': 'm.png',
                                                    'days': 'd.png'})
        assert readme.index('| 2024-03 |') < readme.index('| 2024-02 |') < readme.index('| 2024-01 |')
        page = yb.render_quarter('2024Q1', {}, stats)
        assert page.index('| 2024-03 |') < page.index('| 2024-01 |')
        assert page.index('| 2024-03-01 |') < page.index('| 2024-01-01 |')


@pytest.mark.unittest
class TestRepeatAudit:
    def test_a_file_in_two_packs_is_reported_newest_first(self):
        packs = {
            'pack_20261007_100000_000001.zip': [('0GRM_3_a.png', 10), ('x.png', 1)],
            'pack_20261008_100000_000001.zip': [('0GRM_3_a.png', 10), ('y.png', 2)],
            'pack_20261008_200000_000001.zip': [('0GRM_3_a.png', 11)],  # same name, other size: another version
        }
        repeated = yb.find_repeated_files(packs)
        assert repeated == [('0GRM_3_a.png', 10, ['pack_20261008_100000_000001.zip', 'pack_20261007_100000_000001.zip'])]

    def test_only_the_last_days_are_audited(self, monkeypatch):
        class Item:
            def __init__(self, path, size):
                self.path, self.size = path, size

        class Api:
            def list_repo_tree(self, **kwargs):
                return [Item('packs/pack_20261001_100000_000001.zip', 1), Item('packs/pack_20261008_100000_000001.zip', 1),
                        Item('packs/pack_20261007_100000_000001.zip', 1)]
        read = []

        def fake_read(url, token, size):
            read.append(url)
            return [('same.png', 5)]
        monkeypatch.setattr(yb, 'read_members_remote', fake_read)
        now = dt.datetime(2026, 10, 8, 12, 0, tzinfo=yb.LOCAL_TZ)
        repeated = yb.audit_recent(Api(), 'hk1901/fuck_the_skeb', None, days=3, now=now)
        assert len(read) == 2 and not any('20261001' in url for url in read)
        assert repeated == [('same.png', 5, ['pack_20261008_100000_000001.zip', 'pack_20261007_100000_000001.zip'])]
