import pytest

from test.prepare.errors import ResourceGone, ResourceTransient, UnexpectedResponse
from test.prepare.sites import imgur


@pytest.mark.unittest
class TestDirectLinks:
    def _patch(self, monkeypatch, fetched, api_result=None, fetch_error=None):
        def fake_fetch(fx, url, out_dir, name=None, **kwargs):
            fetched.append(url)
            if fetch_error and not url.startswith('https://cdn.example'):
                raise fetch_error
            return url

        monkeypatch.setattr(imgur, 'fetch_file', fake_fetch)
        monkeypatch.setattr(imgur, '_api', lambda fx, path: api_result)

    def test_link_without_extension_goes_through_the_media_api(self, monkeypatch):
        fetched = []
        self._patch(monkeypatch, fetched, {'media': [{'url': 'https://cdn.example/real.png', 'name': 'real.png'}]})
        imgur.download(None, 'https://i.imgur.com/Gq5lCgM', '/tmp/x')
        assert fetched == ['https://cdn.example/real.png']  # the html wrapper itself was never requested

    def test_html_answer_for_a_link_with_extension_falls_back_to_the_api(self, monkeypatch):
        fetched = []
        self._patch(monkeypatch, fetched, {'media': [{'url': 'https://cdn.example/real.png', 'name': 'real.png'}]},
                    fetch_error=UnexpectedResponse(200, 'text/html', '<html>', 'u'))
        imgur.download(None, 'https://i.imgur.com/Gq5lCgM.png', '/tmp/x')
        assert fetched == ['https://i.imgur.com/Gq5lCgM.png', 'https://cdn.example/real.png']

    def test_other_http_errors_stay_retryable(self, monkeypatch):
        self._patch(monkeypatch, [], None, fetch_error=UnexpectedResponse(503, 'text/html', '', 'u'))
        with pytest.raises(ResourceTransient):
            imgur.download(None, 'https://i.imgur.com/Gq5lCgM.png', '/tmp/x')

    def test_unknown_media_id_is_gone(self, monkeypatch):
        self._patch(monkeypatch, [], None)
        with pytest.raises(ResourceGone):
            imgur.download(None, 'https://i.imgur.com/Gq5lCgM', '/tmp/x')
