import pytest

from test.prepare.errors import ResourceGone, NoContent, ResourceTransient
from test.prepare.sites import pixiv


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class Fx:
    def __init__(self, routes, heads=()):
        self.routes = routes
        self.heads = set(heads)

    def get(self, url, **kwargs):
        return self.routes[url.rsplit('/ajax/illust/', 1)[1]]

    def head(self, url, **kwargs):
        return Resp(200 if url in self.heads else 404, {})


THUMB = 'https://i.pximg.net/c/250x250/img-master/img/2023/05/06/07/08/09/555_p0_square1200.jpg'


@pytest.mark.unittest
class TestPixivResolution:
    def test_all_ages_work_uses_the_pages_endpoint(self):
        fx = Fx({'555/pages': Resp(200, {'error': False, 'body': [{'urls': {'original': 'https://i.pximg.net/a_p0.png'}}]})})
        assert pixiv._original_urls(fx, '555') == ['https://i.pximg.net/a_p0.png']

    def test_r18_work_is_rebuilt_from_the_thumbnail_path(self):
        original = 'https://i.pximg.net/img-original/img/2023/05/06/07/08/09/555_p'
        fx = Fx({
            '555/pages': Resp(404, {'error': True, 'message': ''}),  # what pixiv answers logged out for R-18
            '555': Resp(200, {'error': False, 'body': {'pageCount': 2, 'userIllusts': {'555': {'url': THUMB}}}}),
        }, heads={original + '0.png'})
        assert pixiv._original_urls(fx, '555') == [original + '0.png', original + '1.png']

    def test_deleted_work_is_gone_only_when_both_endpoints_say_so(self):
        fx = Fx({'555/pages': Resp(404, {'error': True}), '555': Resp(404, {'error': True, 'message': 'deleted'})})
        with pytest.raises(ResourceGone):
            pixiv._original_urls(fx, '555')

    def test_missing_thumbnail_is_no_content_not_gone(self):
        fx = Fx({'555/pages': Resp(404, {'error': True}), '555': Resp(200, {'error': False, 'body': {}})})
        with pytest.raises(NoContent):
            pixiv._original_urls(fx, '555')

    def test_unreachable_rebuilt_url_is_no_content(self):
        fx = Fx({'555/pages': Resp(404, {'error': True}),
                 '555': Resp(200, {'error': False, 'body': {'pageCount': 1, 'userIllusts': {'555': {'url': THUMB}}}})})
        with pytest.raises(NoContent):
            pixiv._original_urls(fx, '555')

    def test_garbage_answers_are_retryable(self):
        fx = Fx({'555/pages': Resp(200, ValueError('no json'))})
        with pytest.raises(ResourceTransient):
            pixiv._original_urls(fx, '555')
