"""Statistics pages of the dataset: an overview README, one index page per quarter and the chart images.

Only markdown pages and PNG charts are written here. The file list of a pack is read from its zip central
directory with HTTP range requests, so no pack is ever downloaded. Time buckets come from the pack names, which
the repacker writes in UTC; they are shown in UTC+8 and mean the repack time, not the time a file was crawled.
"""
import datetime as dt
import gzip
import io
import json
import os
import re
import struct
import tempfile
import zipfile
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

PACK_RE = re.compile(r'^pack_(\d{8})_(\d{6})_\d+\.zip$')
LOCAL_TZ = dt.timezone(dt.timedelta(hours=8))
REPO_URL = 'https://hub.deepghs.org/datasets/hk1901/fuck_the_skeb'

# big categories of the files a user wants to know about, by extension
CATEGORIES: Dict[str, Tuple[str, ...]] = {
    'image': ('png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff', 'heic', 'heif', 'avif', 'jfif'),
    'animated': ('gif', 'apng'),
    'layered': ('psd', 'psb', 'clip', 'csp', 'procreate', 'kra', 'ora', 'xcf', 'pdn', 'sai', 'sai2', 'ai', 'afdesign'),
    'video': ('mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v', 'wmv', 'flv'),
    'audio': ('mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'opus', 'wma'),
    'model': ('fbx', 'obj', 'blend', 'vrm', 'pmx', 'pmd', 'glb', 'gltf', 'moc3', 'vmd'),
    'document': ('pdf', 'txt', 'doc', 'docx', 'rtf', 'md', 'odt', 'xls', 'xlsx', 'csv', 'pptx'),
    'archive': ('zip', 'rar', '7z', 'tar', 'gz', 'tgz', 'bz2', 'xz'),
}
EXT_TO_CATEGORY = {ext: cat for cat, exts in CATEGORIES.items() for ext in exts}
CATEGORY_ORDER = list(CATEGORIES) + ['other']


def category_of(name: str) -> str:
    base = name.rsplit('/', 1)[-1]
    if '.' not in base:
        return 'other'
    return EXT_TO_CATEGORY.get(base.rsplit('.', 1)[-1].lower(), 'other')


def pack_time(name: str) -> Optional[dt.datetime]:
    """Repack time of a pack (UTC in the name), shown in UTC+8. None for names that do not follow the pattern."""
    match = PACK_RE.match(name)
    if match is None:
        return None
    stamp = dt.datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
    return stamp.replace(tzinfo=dt.timezone.utc).astimezone(LOCAL_TZ)


GetRange = Callable[[int, int], bytes]


def read_members(size: int, get_range: GetRange) -> List[Tuple[str, int]]:
    """Return ``(name, uncompressed size)`` of every file in a zip, reading only its end and central directory."""
    tail_start = max(0, size - 65557)
    tail = get_range(tail_start, size - 1)
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0:
        raise ValueError('no end of central directory record')
    count, _, cd_offset = struct.unpack('<HII', tail[pos + 10:pos + 20])
    cd_size = struct.unpack('<I', tail[pos + 12:pos + 16])[0]
    if count == 0xFFFF or cd_offset == 0xFFFFFFFF:  # zip64: the real numbers are in the zip64 end record
        locator = tail[pos - 20:pos]
        if locator[:4] != b'PK\x06\x07':
            raise ValueError('zip64 locator missing')
        record_offset = struct.unpack('<Q', locator[8:16])[0]
        record = get_range(record_offset, record_offset + 55)
        count, cd_size, cd_offset = struct.unpack('<QQQ', record[32:56])
    cd = get_range(cd_offset, cd_offset + cd_size - 1)
    members, p = [], 0
    while p < len(cd):
        if cd[p:p + 4] != b'PK\x01\x02':
            raise ValueError('bad central directory entry')
        flags = struct.unpack('<H', cd[p + 8:p + 10])[0]
        usize = struct.unpack('<I', cd[p + 24:p + 28])[0]
        name_len, extra_len, comment_len = struct.unpack('<HHH', cd[p + 28:p + 34])
        name = cd[p + 46:p + 46 + name_len].decode('utf-8' if flags & 0x800 else 'cp437', 'replace')
        if usize == 0xFFFFFFFF:  # the zip64 extra field carries the real size
            extra = cd[p + 46 + name_len:p + 46 + name_len + extra_len]
            q = 0
            while q + 4 <= len(extra):
                tag, length = struct.unpack('<HH', extra[q:q + 4])
                if tag == 0x0001 and length >= 8:
                    usize = struct.unpack('<Q', extra[q + 4:q + 12])[0]
                    break
                q += 4 + length
        if not name.endswith('/'):
            members.append((name, usize))
        p += 46 + name_len + extra_len + comment_len
    return members


def summarize_pack(name: str, size: int, members: List[Tuple[str, int]]) -> Dict:
    """One record per pack: the numbers the overview and the monthly charts need."""
    cats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for member, usize in members:
        cat = category_of(member)
        cats[cat][0] += 1
        cats[cat][1] += usize
    stamp = pack_time(name)
    return {'name': name, 'time': stamp.isoformat() if stamp else None, 'size': size,
            'files': len(members), 'bytes': sum(usize for _, usize in members),
            'categories': {cat: values for cat, values in sorted(cats.items())}}


# ---------------------------------------------------------------- aggregation
def _quarter_key(stamp: dt.datetime) -> str:
    return f'{stamp.year}Q{(stamp.month - 1) // 3 + 1}'


def aggregate(records: List[Dict]) -> Dict:
    """Totals per category, per quarter, per month and per day, from the pack records."""
    dated = [r for r in records if r['time']]
    dated.sort(key=lambda r: r['time'])
    totals = {'packs': len(records), 'files': 0, 'bytes': 0, 'categories': Counter(), 'category_bytes': Counter()}
    by_quarter: Dict[str, Dict] = {}
    by_month: Dict[str, Dict] = {}
    by_day: Dict[str, Dict] = {}
    for record in records:
        totals['files'] += record['files']
        totals['bytes'] += record['bytes']
        for cat, (count, nbytes) in record['categories'].items():
            totals['categories'][cat] += count
            totals['category_bytes'][cat] += nbytes
    for record in dated:
        stamp = dt.datetime.fromisoformat(record['time'])
        for table, key in ((by_quarter, _quarter_key(stamp)), (by_month, stamp.strftime('%Y-%m')),
                           (by_day, stamp.strftime('%Y-%m-%d'))):
            bucket = table.setdefault(key, {'packs': 0, 'files': 0, 'bytes': 0, 'categories': Counter(),
                                            'category_bytes': Counter()})
            bucket['packs'] += 1
            bucket['files'] += record['files']
            bucket['bytes'] += record['bytes']
            for cat, (count, nbytes) in record['categories'].items():
                bucket['categories'][cat] += count
                bucket['category_bytes'][cat] += nbytes
    first = dated[0]['time'] if dated else None
    last = dated[-1]['time'] if dated else None
    return {'totals': totals, 'quarters': by_quarter, 'months': by_month, 'days': by_day,
            'first': first, 'last': last, 'dated': dated}


def human_bytes(value: float) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{value:.1f} TiB'


def _share(part: float, whole: float) -> str:
    return f'{100.0 * part / whole:.1f}%' if whole else '0.0%'


# ---------------------------------------------------------------- markdown helpers
def _cell(text: str) -> str:
    return text.replace('\\', '\\\\').replace('|', '\\|').replace('[', '\\[').replace(']', '\\]').replace('\n', ' ')


def _table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def pack_url(pack: str) -> str:
    """The file detail page of a pack in the dataset repository (not the download link)."""
    return f'{REPO_URL}/blob/main/packs/{pack}'


def quarter_url(quarter: str) -> str:
    return f'{REPO_URL}/blob/main/index/{quarter}.md'


def _category_row(counts: Counter, nbytes: Counter, total_files: int, total_bytes: int) -> List[List[str]]:
    rows = []
    for cat in CATEGORY_ORDER:
        if counts[cat] == 0:
            continue
        rows.append([cat, f'{counts[cat]:,}', _share(counts[cat], total_files),
                     human_bytes(nbytes[cat]), _share(nbytes[cat], total_bytes)])
    return rows


# ---------------------------------------------------------------- charts
def render_charts(stats: Dict, directory: str) -> Dict[str, str]:
    """Write the PNG charts into ``directory`` and return their file names."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    written = {}
    colors = plt.get_cmap('tab10')
    quarters = sorted(stats['quarters'])

    def stacked(ax, labels, series_by_label, title):
        bottoms = [0] * len(labels)
        for index, cat in enumerate(CATEGORY_ORDER):
            values = [series_by_label[label].get(cat, 0) for label in labels]
            if not any(values):
                continue
            ax.bar(labels, values, bottom=bottoms, label=cat, color=colors(index % 10))
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_title(title)
        ax.set_ylabel('files')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8, ncol=3)

    # 1. every quarter, stacked by category
    fig, ax = plt.subplots(figsize=(10, 4.5))
    stacked(ax, quarters, {q: stats['quarters'][q]['categories'] for q in quarters}, 'Files per quarter')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'quarters.png'), dpi=110)
    plt.close(fig)
    written['quarters'] = 'quarters.png'

    # 2. the last twelve months: files by category, bytes as a line
    months = sorted(stats['months'])[-12:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    stacked(ax, months, {m: stats['months'][m]['categories'] for m in months}, 'Files per month, last 12 months')
    ax2 = ax.twinx()
    ax2.plot(months, [stats['months'][m]['bytes'] / 1024 ** 3 for m in months], color='black', marker='o')
    ax2.set_ylabel('GiB')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'months_12.png'), dpi=110)
    plt.close(fig)
    written['months'] = 'months_12.png'

    # 3. the last thirty days: files per day
    days = sorted(stats['days'])[-30:]
    fig, ax = plt.subplots(figsize=(10, 4))
    stacked(ax, days, {d: stats['days'][d]['categories'] for d in days}, 'Files per day, last 30 days')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'days_30.png'), dpi=110)
    plt.close(fig)
    written['days'] = 'days_30.png'

    # 4. overall composition by category
    totals = stats['totals']
    cats = [c for c in CATEGORY_ORDER if totals['categories'][c]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cats, [totals['categories'][c] for c in cats], color=[colors(i % 10) for i in range(len(cats))])
    ax.invert_yaxis()
    ax.set_xlabel('files')
    ax.set_title('Files by category, all packs')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, 'categories.png'), dpi=110)
    plt.close(fig)
    written['categories'] = 'categories.png'
    return written


# ---------------------------------------------------------------- pages
def render_quarter(quarter: str, packs_by_month: Dict[str, List[Dict]], stats: Dict) -> str:
    """The index page of one quarter: one row per zip pack, grouped by month."""
    q = stats['quarters'].get(quarter, {'packs': 0, 'files': 0, 'bytes': 0, 'categories': Counter(),
                                        'category_bytes': Counter()})
    out = [f'# Index {quarter}', '',
           f'Every pack whose repack time falls in {quarter} (UTC+8). One row per zip; the name links to the file page '
           f'of the pack. Back to the [overview]({REPO_URL}/blob/main/README.md).', '',
           f'**{q["files"]:,} files**, {human_bytes(q["bytes"])}, in {q["packs"]} packs.', '',
           '## Summary by category', '',
           _table(['Category', 'Files', 'Share', 'Size', 'Share'],
                  _category_row(q['categories'], q['category_bytes'], q['files'], q['bytes'])
                  or [['-', '0', '-', '-', '-']]),
           '', '## Summary by month', '']
    month_rows = []
    for month in sorted((m for m in stats['months'] if _quarter_key(dt.datetime.strptime(m, '%Y-%m')) == quarter),
                        reverse=True):
        m = stats['months'][month]
        month_rows.append([month, str(m['packs']), f'{m["files"]:,}', human_bytes(m['bytes'])])
    out += [_table(['Month', 'Packs', 'Files', 'Size'], month_rows or [['-', '0', '0', '-']]), '',
            '## Summary by day', '']
    day_rows = []
    for day in sorted((d for d in stats['days']
                       if _quarter_key(dt.datetime.strptime(d, '%Y-%m-%d')) == quarter), reverse=True):
        d = stats['days'][day]
        day_rows.append([day, str(d['packs']), f'{d["files"]:,}', human_bytes(d['bytes'])])
    out += [_table(['Day', 'Packs', 'Files', 'Size'], day_rows or [['-', '0', '0', '-']]), '']

    refs = []
    for month in sorted(packs_by_month, reverse=True):  # newest month first
        headers = ['Pack (repack time)', 'Zip size', 'Content', 'Files'] + CATEGORY_ORDER
        out += [f'## {month}', '', '| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
        for record in sorted(packs_by_month[month], key=lambda r: r['time'], reverse=True):  # newest pack first
            refs.append(record['name'])
            stamp = dt.datetime.fromisoformat(record['time']).strftime('%m-%d %H:%M')
            counts = [str(record['categories'].get(cat, [0, 0])[0]) for cat in CATEGORY_ORDER]
            out.append('| ' + ' | '.join([f'[{record["name"]}][{pack_ref(record["name"])}] ({stamp})',
                                         human_bytes(record['size']), human_bytes(record['bytes']),
                                         f'{record["files"]:,}', *counts]) + ' |')
        out.append('')
    for pack in dict.fromkeys(refs):
        out.append(f'[{pack_ref(pack)}]: {pack_url(pack)}')
    return '\n'.join(out) + '\n'


def pack_ref(pack: str) -> str:
    """A short markdown reference label for a pack: ``pack_20240501_125415_430239.zip`` -> ``p20240501125415``."""
    match = PACK_RE.match(pack)
    return 'p' + (match.group(1) + match.group(2) if match else re.sub(r'\W', '', pack))


def quarter_months_prefix(quarter: str) -> str:
    year, number = int(quarter[:4]), int(quarter[-1])
    first = (number - 1) * 3 + 1
    return f'{year}-{first:02d}'


def render_readme(stats: Dict, quarters_written: List[str], charts: Dict[str, str]) -> str:
    totals = stats['totals']
    first = stats['first'][:10] if stats['first'] else '-'
    last = stats['last'][:10] if stats['last'] else '-'
    out = ['---', 'license: other', '---', '',
           '# fuck_the_skeb: dataset overview', '',
           'Files extracted from Skeb.jp commission posts, packed into zip files under `packs/`. This page is '
           'generated by the repacker together with the quarterly index pages.', '',
           '## Overview', '',
           _table(['Item', 'Value'], [
               ['Files', f'{totals["files"]:,}'],
               ['Size (uncompressed)', human_bytes(totals['bytes'])],
               ['Packs', f'{totals["packs"]:,}'],
               ['First repack', first],
               ['Last repack', last],
           ]), '', '## Categories', '',
           _table(['Category', 'Files', 'Share', 'Size', 'Share'],
                  _category_row(totals['categories'], totals['category_bytes'], totals['files'], totals['bytes'])),
           '', f'![categories](./stats/{charts["categories"]})', '',
           '## Yearbook', '',
           'Every file, grouped by the quarter of its pack\'s repack time. Each page has monthly and daily summaries '
           'and the complete file list with links to the pack.', '',
           _table(['Quarter', 'Packs', 'Files', 'Size', 'Index'], [
               [q, str(stats['quarters'][q]['packs']), f'{stats["quarters"][q]["files"]:,}',
                human_bytes(stats['quarters'][q]['bytes']), f'[{q}]({quarter_url(q)})']
               for q in sorted(stats['quarters'], reverse=True)
           ]), '',
           '## All quarters', '', f'![quarters](./stats/{charts["quarters"]})', '',
           '## Last twelve months', '', f'![months](./stats/{charts["months"]})', '',
           _table(['Month', 'Packs', 'Files', 'Size'], [
               [m, str(stats['months'][m]['packs']), f'{stats["months"][m]["files"]:,}',
                human_bytes(stats['months'][m]['bytes'])] for m in sorted(stats['months'], reverse=True)[:12]
           ]), '',
           '## Last thirty days', '', f'![days](./stats/{charts["days"]})', '',
           _table(['Day', 'Packs', 'Files', 'Size'], [
               [d, str(stats['days'][d]['packs']), f'{stats["days"][d]["files"]:,}',
                human_bytes(stats['days'][d]['bytes'])] for d in sorted(stats['days'], reverse=True)[:30]
           ]), '',
           '## Method', '',
           '- Files are listed from each zip\'s central directory with HTTP range requests; no pack is downloaded to '
           'count them.',
           '- Time buckets use the repack time in the pack name (UTC, shown as UTC+8). A file is counted in the '
           'bucket of the pack that contains it, which can be a few hours after it was crawled.',
           '- Categories are by file extension. Files without a known extension are counted as other.',
           '- The file names in the indexes link to the pack file page, where the zip can be opened.', '']
    return '\n'.join(out)


# ---------------------------------------------------------------- data files
MANIFEST_PATH = 'stats/packs.json'  # one summary record per pack; every page is rendered from it


def zip_members_local(path: str) -> List[Tuple[str, int]]:
    """The members of a zip that is on the local disk (the pack that was just built)."""
    with zipfile.ZipFile(path) as zf:
        return [(info.filename, info.file_size) for info in zf.infolist() if not info.filename.endswith('/')]


# ---------------------------------------------------------------- building
def render_pages(records: List[Dict], out_dir: str, quarters: List[str]) -> List[str]:
    """Write the index pages for ``quarters``, the charts, README.md and the manifest into ``out_dir``."""
    stats = aggregate(records)
    written = []
    os.makedirs(os.path.join(out_dir, 'index'), exist_ok=True)
    for quarter in sorted(set(quarters) & set(stats['quarters'])):
        packs_by_month: Dict[str, List[Dict]] = defaultdict(list)
        for record in stats['dated']:
            stamp = dt.datetime.fromisoformat(record['time'])
            if _quarter_key(stamp) == quarter:
                packs_by_month[stamp.strftime('%Y-%m')].append(record)
        rel = f'index/{quarter}.md'
        with open(os.path.join(out_dir, rel), 'w', encoding='utf-8') as f:
            f.write(render_quarter(quarter, packs_by_month, stats))
        written.append(rel)
    charts = render_charts(stats, os.path.join(out_dir, 'stats'))
    written += [f'stats/{name}' for name in charts.values()]
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(render_readme(stats, sorted(stats['quarters']), charts))
    written.append('README.md')
    with open(os.path.join(out_dir, MANIFEST_PATH), 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    written.append(MANIFEST_PATH)
    return written


def plan_update(known: List[Dict], new_records: List[Dict]) -> Dict[str, bytes]:
    """The files that change when ``new_records`` are added: only the pages of the quarters that got packs."""
    records = known + new_records
    touched = sorted({quarter_of_pack(record['name']) for record in new_records} - {None})
    files: Dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as td:
        for rel in render_pages(records, td, touched):
            with open(os.path.join(td, rel), 'rb') as f:
                files[rel] = f.read()
    return files


def quarter_of_pack(name: str) -> Optional[str]:
    stamp = pack_time(name)
    return _quarter_key(stamp) if stamp else None


def range_reader(url: str, token: Optional[str]) -> GetRange:
    """A ``get_range`` for one file of the repository, through HTTP Range requests."""
    import requests

    def get(start: int, end: int) -> bytes:
        headers = {'Range': f'bytes={start}-{end}'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        response = requests.get(url, headers=headers, timeout=180)
        if response.status_code not in (200, 206):
            raise RuntimeError(f'range {start}-{end} of {url} -> HTTP {response.status_code}')
        return response.content
    return get


def load_manifest(api, repo_id: str, token: Optional[str]) -> List[Dict]:
    """The pack records the statistics were built from (empty before the first build)."""
    from huggingface_hub import hf_hub_download
    if not api.file_exists(repo_id=repo_id, repo_type='dataset', filename=MANIFEST_PATH):
        return []
    with open(hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=MANIFEST_PATH, token=token),
              encoding='utf-8') as f:
        return json.load(f)


def operations_for(files: Dict[str, bytes]):
    from huggingface_hub import CommitOperationAdd
    return [CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in sorted(files.items())]


def refresh(api, repo_id: str, token: Optional[str]) -> Optional[str]:
    """Catch-up: add every pack the statistics do not know yet, in one commit. None when nothing was new."""
    from huggingface_hub import hf_hub_url

    known = load_manifest(api, repo_id, token)
    names = {record['name'] for record in known}
    packs = [item for item in api.list_repo_tree(repo_id=repo_id, repo_type='dataset', path_in_repo='packs')
             if item.path.endswith('.zip') and os.path.basename(item.path) not in names]
    if not packs:
        return None
    new_records = []
    for item in packs:
        url = hf_hub_url(repo_id=repo_id, repo_type='dataset', filename=item.path)
        members = read_members_remote(url, token, item.size)
        new_records.append(summarize_pack(os.path.basename(item.path), item.size, members))
    message = f'[stats] catch-up for {len(packs)} pack(s) | overview and index pages'
    api.create_commit(repo_id=repo_id, repo_type='dataset', operations=operations_for(plan_update(known, new_records)),
                      commit_message=message,
                      commit_description='Packs: ' + ', '.join(sorted(os.path.basename(i.path) for i in packs)))
    return message


def read_members_remote(url: str, token: Optional[str], size: int) -> List[Tuple[str, int]]:
    return read_members(size, range_reader(url, token))
