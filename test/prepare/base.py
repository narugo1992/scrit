import math
import os

from PIL import Image
from hfutils.index import tar_get_index_info
from huggingface_hub import HfApi, HfFileSystem, configure_http_backend

from pyskeb.utils import get_requests_session

Image.MAX_IMAGE_PIXELS = 17000 ** 2
configure_http_backend(get_requests_session)

hf_token = os.environ.get('HF_TOKEN')
hf_client = HfApi(token=hf_token)
hf_fs = HfFileSystem(token=hf_token)

_REPOSITORY = os.environ['REMOTE_REPOSITORY']


class GenericException(Exception):
    pass


def _ensure_repository():
    if not hf_client.repo_exists(repo_id=_REPOSITORY, repo_type='dataset'):
        hf_client.create_repo(
            repo_id=_REPOSITORY,
            repo_type='dataset',
            exist_ok=True,
            private=True,
        )
        lines = hf_fs.read_text(f'datasets/{_REPOSITORY}/.gitattributes').splitlines(keepends=False)
        lines = [*filter(bool, lines), 'archived.json filter=lfs diff=lfs merge=lfs -text']
        hf_fs.write_text(f'datasets/{_REPOSITORY}/.gitattributes', os.linesep.join(lines))


_NUM_TAGS = [
    ('n<1K', 0, 1_000),
    ('1K<n<10K', 1_000, 10_000),
    ('10K<n<100K', 10_000, 100_000),
    ('100K<n<1M', 100_000, 1_000_000),
    ('1M<n<10M', 1_000_000, 10_000_000),
    ('10M<n<100M', 10_000_000, 100_000_000),
    ('100M<n<1B', 100_000_000, 1_000_000_000),
    ('1B<n<10B', 1_000_000_000, 10_000_000_000),
    ('10B<n<100B', 10_000_000_000, 100_000_000_000),
    ('100B<n<1T', 100_000_000_000, 1_000_000_000_000),
    ('n>1T', 1_000_000_000_000, math.inf),
]


def number_to_tag(v):
    for tag, min_, max_ in _NUM_TAGS:
        if min_ <= v < max_:
            return tag

    raise ValueError(f'No tags found for {v!r}')


def make_index_file(src_tar_file, chunk_for_hash: int = 1 << 20):
    return tar_get_index_info(src_tar_file, chunk_for_hash, with_hash=True)
