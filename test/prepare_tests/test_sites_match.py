import pytest

from test.prepare.sites import resolve
from test.prepare.url import extract_urls


def rid(url):
    resolved = resolve(url)
    return resolved[1] if resolved else None


@pytest.mark.unittest
class TestExtractUrls:
    def test_cut_at_japanese_text(self):
        text = 'ref https://drive.google.com/file/d/1BazF3S2h121xpwnkvjiWxhItu6WfMPgw/view?usp=drive_link彼女は小柄で 以上'
        assert extract_urls(text) == ['https://drive.google.com/file/d/1BazF3S2h121xpwnkvjiWxhItu6WfMPgw/view?usp=drive_link']

    def test_trailing_punctuation_and_dedupe(self):
        text = 'see https://x.com/a/status/1. and https://x.com/a/status/1, ok'
        assert extract_urls(text) == ['https://x.com/a/status/1']

    def test_empty(self):
        assert extract_urls(None) == []
        assert extract_urls('no link here') == []


@pytest.mark.unittest
class TestResourceIds:
    @pytest.mark.parametrize('url, expected', [
        ('https://drive.google.com/drive/folders/1KXzKjL23zSvD0VDyujfvjuTD-9oMWiot', 'googledrive_1KXzKjL23zSvD0VDyujfvjuTD-9oMWiot'),
        ('https://drive.google.com/drive/folders/1K3OP7Pq9HcL28jkeQfaFCAp0T3139VVV?usp=sharing', 'googledrive_1K3OP7Pq9HcL28jkeQfaFCAp0T3139VVV'),
        ('https://drive.google.com/drive/u/0/mobile/folders/1jlILAUSkORjDDbTrArPhn42ANUlG3sK9', 'googledrive_1jlILAUSkORjDDbTrArPhn42ANUlG3sK9'),
        ('https://drive.google.com/drive/u/1/folders/1Lg9jDJs05UgXFUx_tNJv4hIGq-w4WJBq', 'googledrive_1Lg9jDJs05UgXFUx_tNJv4hIGq-w4WJBq'),
        ('https://drive.google.com/file/d/16kD8AfiaTN6HpmibK1APnGzWGaPNPwqR/view?usp=drivesdk', 'googledrive_16kD8AfiaTN6HpmibK1APnGzWGaPNPwqR'),
        ('https://drive.google.com/open?id=1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa', 'googledrive_1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa'),
        ('https://drive.google.com/uc?id=1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa&export=download', 'googledrive_1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa'),
        ('https://docs.google.com/document/d/1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa/edit', 'googledrive_1ocFPNA0ZOjPxtKJFv-DFJDj4M0XLuBXa'),
        ('https://drive.google.com/drive/folders/1', None),
        ('https://imgur.com/a/hsDA3kQ', 'imgur_hsDA3kQ'),
        ('https://imgur.com/a/20251230-ZBnEC2F', 'imgur_ZBnEC2F'),
        ('https://imgur.com/gallery/some-title-AbCdEfG', 'imgur_AbCdEfG'),
        ('https://imgur.com/AbCdEfG', 'imgur_media_AbCdEfG'),
        ('https://i.imgur.com/AbCdEfG.png', 'imgur_media_AbCdEfG'),
        ('https://imgur.com/upload', None),
        # the Dropbox id rule must not change: it is part of the archive of ~10k resources
        ('https://www.dropbox.com/scl/fi/fkuzsfifyusrnfqrhg3ko/image0-1.png?rlkey=3azmwutcq0qys07m0v438kni6&st=dcm28rfi&dl=0',
         'dropbox_scl_fi_fkuzsfifyusrnfqrhg3ko_image0-1.png'),
        ('https://www.dropbox.com/scl/fo/hdpoeyxj9v3d6hltlnjil/AIsRz7dv_N67UPxcuGKVyc8?rlkey=x&dl=0',
         'dropbox_scl_fo_hdpoeyxj9v3d6hltlnjil_AIsRz7dv_N67UPxcuGKVyc8'),
        ('https://www.dropbox.com/s/087nlo8izrmgkni/%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88%E8%B3%87%E6%96%99.jpg?dl=0',
         'dropbox_s_087nlo8izrmgkni_イラスト資料.jpg'),
        ('https://www.dropbox.com/sh/r2btg18v0990hx2/AABeRc55Y0k1ubKS8HAaFcu-a?dl=0', 'dropbox_sh_r2btg18v0990hx2_AABeRc55Y0k1ubKS8HAaFcu-a'),
        ('https://www.dropbox.com/home', None),
        ('https://x.com/someone/status/1664436217696636928?s=20', 'twitter_1664436217696636928'),
        ('https://twitter.com/i/web/status/1664436217696636928', 'twitter_1664436217696636928'),
        ('https://x.com/someone', None),
        ('https://pbs.twimg.com/media/GAbcdef123?format=jpg&name=large', 'twimg_GAbcdef123'),
        ('https://www.pixiv.net/artworks/118586503', 'pixiv_118586503'),
        ('https://www.pixiv.net/en/artworks/118586503', 'pixiv_118586503'),
        ('https://www.pixiv.net/member_illust.php?mode=medium&illust_id=555', 'pixiv_555'),
        ('https://www.pixiv.net/users/1', None),
        ('https://files.catbox.moe/abc123.png', 'catbox_abc123'),
        ('https://catbox.moe/c/xyz789', 'catbox_album_xyz789'),
        ('https://imgchest.com/p/n87wazajg4x', 'imgchest_n87wazajg4x'),
        ('https://gyazo.com/65bc7801f790fb8b45048ce3843118bc', 'gyazo_65bc7801f790fb8b45048ce3843118bc'),
        ('https://i.gyazo.com/65bc7801f790fb8b45048ce3843118bc.jpg', 'gyazo_65bc7801f790fb8b45048ce3843118bc'),
        ('https://ibb.co/LXtn9FP4', 'ibb_LXtn9FP4'),
        ('https://ibb.co/album/HtrSmd', 'ibb_album_HtrSmd'),
        ('https://ibb.co/page', None),
        ('https://postimg.cc/rwK23Rj5', 'postimg_rwK23Rj5'),
        ('https://postimg.cc/gallery/Qw1ErTy', 'postimg_gallery_Qw1ErTy'),
        ('https://photos.app.goo.gl/AbCdEf123456', 'gphotos_AbCdEf123456'),
        ('https://bsky.app/profile/someone.bsky.social/post/3kabc', 'bsky_someone_bsky_social_3kabc'),
        ('https://misskey.io/notes/9abcdef', 'misskey_note_9abcdef'),
        ('https://misskey.io/@user/pages/my-page', 'misskey_page_user_my-page'),
        ('https://misskey.io/clips/abc123', 'misskey_clip_abc123'),
        ('https://skeb.jp/@someone/works/1', None),
        ('https://www.youtube.com/watch?v=abc', None),
    ])
    def test_resolve(self, url, expected):
        assert rid(url) == expected

    def test_resource_ids_are_filename_safe(self):
        for url in ['https://bsky.app/profile/a.b.c/post/xyz', 'https://misskey.io/@u.v/pages/p.q']:
            value = rid(url)
            assert '.' not in value and '/' not in value
