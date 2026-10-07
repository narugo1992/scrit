import re
from typing import List

import xurls

_TRAILING = '.,;:!?\'"`'
_NON_ASCII = re.compile(r'[^\x00-\x7f]')


def extract_urls(text) -> List[str]:
    """Find http(s) URLs in a post body.

    Japanese text is often glued right after a URL without any whitespace, so every candidate is cut at
    its first non-ASCII character. Otherwise the tail ends up inside Dropbox ``rlkey`` values or Skeb
    work ids.
    """
    extractors = [
        xurls.StrictScheme('https://'),
        xurls.StrictScheme('http://'),
    ]
    urls = set()
    for extractor in extractors:
        for item in extractor.findall(text or ''):
            item = _NON_ASCII.split(item, maxsplit=1)[0].rstrip(_TRAILING)
            if re.match(r'^https?://[^/]+', item):
                urls.add(item)

    return sorted(urls)
