from typing import List, Optional, Tuple

from . import google, imgur, dropbox, twitter, pixiv, onedrive
from .fediverse import bsky, misskey
from .hosts import catbox, imgchest, gyazo, ibb, postimg, gphotos, direct

# The first site whose ``match`` returns a resource id wins. Order matters only where hosts overlap.
SITES = [google, imgur, dropbox, twitter, pixiv, catbox, imgchest, gyazo, ibb, postimg, gphotos,
         bsky, misskey, onedrive, direct]


def resolve(url: str) -> Optional[Tuple[object, str]]:
    for site in SITES:
        resource_id = site.match(url)
        if resource_id:
            return site, resource_id
    return None


def names() -> List[str]:
    return [site.NAME for site in SITES]
