import pytest

from test.prepare.http import safe_name


@pytest.mark.unittest
class TestSafeName:
    def test_long_japanese_names_fit_the_file_system_and_keep_the_extension(self):
        name = 'ユウビ新ビジュ' * 40 + '.png'
        result = safe_name(name)
        assert len(result.encode('utf-8')) <= 200
        assert result.endswith('.png') and result.startswith('ユウビ')
        result.encode('utf-8').decode('utf-8')  # cut between characters, never inside one

    def test_short_names_are_unchanged(self):
        assert safe_name('絵 1.png') == '絵 1.png'

    def test_a_long_extension_is_not_kept_but_the_name_is_still_cut(self):
        result = safe_name('a' * 300 + '.' + 'x' * 100)
        assert len(result.encode('utf-8')) <= 200

    def test_path_separators_are_still_replaced(self):
        assert safe_name('a/b\\c') == 'a_b_c'
