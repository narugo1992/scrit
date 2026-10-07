from collections import Counter
from typing import Iterable


def pretty_size(size: float) -> str:
    """812 B, 46.1 MiB, 5.5 GiB: never a bare byte count."""
    size = float(size)
    if size < 1024:
        return f'{int(size)} B'
    for unit in ('KiB', 'MiB', 'GiB', 'TiB'):
        size /= 1024.0
        if size < 1024 or unit == 'TiB':
            return f'{size:.1f} {unit}'
    raise AssertionError('unreachable')  # pragma: no cover


def plural(count: int, word: str) -> str:
    return f'{count} {word}' if count == 1 else f'{count} {word}s'


def by_site(names: Iterable[str]) -> str:
    counts = Counter(names)
    return ', '.join(f'{name} {count}' for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
