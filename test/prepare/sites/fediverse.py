"""Social posts that expose their attached images through public APIs (bluesky, misskey)."""
import re
from typing import List, Optional

from .common import host_of, segments, fetch_file, clean_id
from ..errors import ResourceGone, ResourceTransient, NoContent, UnexpectedResponse
from ..http import Fetcher

_BSKY_API = 'https://public.api.bsky.app/xrpc/'


def _fetch_each(fx: Fetcher, items: List[tuple], out_dir: str):
    if not items:
        raise NoContent('no media attached')
    for media_url, name in items:
        try:
            fetch_file(fx, media_url, out_dir, name)
        except UnexpectedResponse as err:
            raise ResourceTransient(str(err)) from err


class bsky:
    NAME = 'bsky'

    @staticmethod
    def parse(url: str):
        if host_of(url) != 'bsky.app':
            return None
        found = re.fullmatch(r'/profile/([^/]+)/post/(\w+)/?', re.sub(r'\?.*$', '', url.split('bsky.app', 1)[-1]))
        return found.groups() if found else None

    @staticmethod
    def match(url: str) -> Optional[str]:
        parsed = bsky.parse(url)
        return f'bsky_{clean_id(parsed[0])}_{parsed[1]}' if parsed else None

    @staticmethod
    def download(fx: Fetcher, url: str, out_dir: str):
        actor, rkey = bsky.parse(url)
        did = actor
        if not actor.startswith('did:'):
            resolved = fx.get(_BSKY_API + 'com.atproto.identity.resolveHandle', params={'handle': actor})
            did = resolved.json().get('did') if resolved.status_code == 200 else None
            if not did:
                raise ResourceGone(f'bsky handle {actor!r} cannot be resolved')
        thread = fx.get(_BSKY_API + 'app.bsky.feed.getPostThread',
                        params={'uri': f'at://{did}/app.bsky.feed.post/{rkey}', 'depth': 0})
        if thread.status_code in (400, 404):
            raise ResourceGone(f'bsky post {url!r} not found')
        if thread.status_code != 200:
            raise ResourceTransient(f'bsky getPostThread -> HTTP {thread.status_code}')
        embed = ((thread.json().get('thread') or {}).get('post') or {}).get('embed') or {}
        images = embed.get('images') or (embed.get('media') or {}).get('images') or []
        if not images:
            raise NoContent(f'bsky post {url!r} has no images')
        did_doc = fx.get(f'https://plc.directory/{did}').json() if did.startswith('did:plc:') else {}
        pds = next((item['serviceEndpoint'] for item in did_doc.get('service') or []
                    if item.get('id') == '#atproto_pds'), None)
        items = []
        for index, image in enumerate(images, start=1):
            cid = image['fullsize'].split('/')[-1].split('@')[0]
            blob = f'{pds}/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}' if pds else image['fullsize']
            items.append((blob, f'{rkey}_{index}.{image["fullsize"].rsplit("@", 1)[-1] if "@" in image["fullsize"] else "jpg"}'))
        _fetch_each(fx, items, out_dir)


class misskey:
    NAME = 'misskey'
    API = 'https://misskey.io/api/'

    @staticmethod
    def parse(url: str):
        if host_of(url) != 'misskey.io':
            return None
        segs = segments(url)
        if len(segs) == 2 and segs[0] == 'notes':
            return 'note', segs[1]
        if len(segs) == 2 and segs[0] == 'clips':
            return 'clip', segs[1]
        if len(segs) == 3 and segs[0].startswith('@') and segs[1] == 'pages':
            return 'page', f'{segs[0][1:]}/{segs[2]}'
        return None

    @staticmethod
    def match(url: str) -> Optional[str]:
        parsed = misskey.parse(url)
        return f'misskey_{parsed[0]}_{clean_id(parsed[1])}' if parsed else None

    @staticmethod
    def _call(fx: Fetcher, endpoint: str, body: dict):
        resp = fx.post(misskey.API + endpoint, json=body)
        if resp.status_code in (400, 404):
            raise ResourceGone(f'misskey {endpoint} {body} not found')
        if resp.status_code != 200:
            raise ResourceTransient(f'misskey {endpoint} -> HTTP {resp.status_code}')
        return resp.json()

    @staticmethod
    def download(fx: Fetcher, url: str, out_dir: str):
        kind, ident = misskey.parse(url)
        if kind == 'note':
            files = misskey._call(fx, 'notes/show', {'noteId': ident}).get('files') or []
        elif kind == 'clip':
            notes = misskey._call(fx, 'clips/notes', {'clipId': ident, 'limit': 100})
            files = [item for note in notes for item in note.get('files') or []]
        else:
            user, name = ident.split('/', 1)
            files = misskey._call(fx, 'pages/show', {'username': user, 'name': name}).get('attachedFiles') or []
        _fetch_each(fx, [(item['url'], item.get('name')) for item in files[:300]], out_dir)
