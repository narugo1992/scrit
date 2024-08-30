import datetime
import mimetypes
import os

import numpy as np
import pandas as pd
from PIL import Image
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_client, get_hf_fs, download_archive_as_directory, upload_directory_as_directory
from hfutils.utils import parse_hf_fs_path, number_to_tag
from tqdm import tqdm

mimetypes.add_type('image/webp', '.webp')


def sync(src_repo: str, dst_repo: str):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='dataset'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='dataset', private=False)
        hf_client.update_repo_visibility(repo_id=dst_repo, repo_type='dataset', private=False)
        attr_lines = hf_fs.read_text(f'datasets/{dst_repo}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{dst_repo}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    src_zip_files = [
        parse_hf_fs_path(file).filename
        for file in hf_fs.glob(f'datasets/{src_repo}/*.zip')
    ]
    src_ids = [os.path.splitext(file)[0] for file in src_zip_files]

    dst_tar_files = [
        os.path.basename(parse_hf_fs_path(file).filename)
        for file in hf_fs.glob(f'datasets/{src_repo}/packs/*.tar')
    ]
    dst_ids = set([os.path.splitext(file)[0] for file in dst_tar_files])

    if hf_fs.exists(f'datasets/{dst_repo}/table.parquet'):
        df = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=dst_repo,
            repo_type='dataset',
            filename='table.parquet'
        )).replace(np.nan, None)
        rows = df.to_dict('records')
        max_id = df['id'].max().item()
    else:
        rows = []
        max_id = 0

    for pack_id in tqdm(src_ids, desc='Sync Packs'):
        if pack_id in dst_ids:
            logging.warning(f'Package {pack_id!r} already synced, skipped.')
            continue

        with TemporaryDirectory() as td:
            tar_file = os.path.join(td, 'packs', f'{pack_id}.tar')
            os.makedirs(os.path.dirname(tar_file), exist_ok=True)

            with TemporaryDirectory() as tmpdir:
                logging.info(f'Downloading {pack_id!r} from src repo ...')
                download_archive_as_directory(
                    repo_id=src_repo,
                    repo_type='dataset',
                    file_in_repo=f'{pack_id}.zip',
                    local_directory=tmpdir,
                )

                logging.info('Packing archive ...')
                archive_pack('tar', tmpdir, tar_file)
                tar_create_index_for_directory(src_tar_directory=td)

                new_row_count = 0
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        filepath = os.path.abspath(os.path.join(root, file))
                        relpath = os.path.relpath(filepath, os.path.abspath(tmpdir))
                        group_name = os.path.dirname(relpath)
                        mimetype, _ = mimetypes.guess_type(relpath)
                        max_id += 1
                        _, ext = os.path.splitext(relpath)
                        filename = f'{max_id}{ext.lower()}'
                        if mimetype and mimetype.startswith('image/'):
                            image = Image.open(filepath)
                            width, height = image.width, image.height
                        else:
                            width, height = None, None

                        rows.append({
                            'id': max_id,
                            'pack_id': pack_id,
                            'archive_file': os.path.relpath(tar_file, td),
                            'file_in_archive': relpath,
                            'group': group_name,
                            'filename': filename,
                            'mimetype': mimetype,
                            'file_size': os.path.getsize(filepath),
                            'width': width,
                            'height': height,
                        })
                        new_row_count += 1

            df = pd.DataFrame(rows)
            df = df.sort_values(by=['id'], ascending=[False])
            df.to_parquet(os.path.join(td, 'table.parquet'), engine='pyarrow', index=False)

            with open(os.path.join(td, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: other', file=f)
                print('task_categories:', file=f)
                print('- image-classification', file=f)
                print('- zero-shot-image-classification', file=f)
                print('- text-to-image', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('- jp', file=f)
                print('tags:', file=f)
                print('- art', file=f)
                print('- anime', file=f)
                print('- not-for-all-audiences', file=f)
                print('size_categories:', file=f)
                print(f'- {number_to_tag(len(df))}', file=f)
                print('---', file=f)
                print('', file=f)

                print(f'# Index Archives', file=f)
                print(f'', file=f)
                current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
                print(f'{plural_word(len(df), "file")} in total. Last updated at `{current_time}.', file=f)
                print(f'', file=f)

                df_hqimage = df[df['group'] == 'hqimage'][
                    ['id', 'group', 'filename', 'mimetype', 'file_size', 'width', 'height', 'archive_file',
                     'file_in_archive']
                ]
                print(f'{plural_word(len(df_hqimage), "image")} with `hqimage` group.', file=f)
                print(f'', file=f)
                print(df_hqimage.to_markdown(index=False), file=f)
                print(f'', file=f)

                for group_name in sorted(set(df['group']) - {'hqimage'}):
                    df_group = df[df['group'] == group_name][
                        ['id', 'group', 'filename', 'mimetype', 'file_size', 'archive_file', 'file_in_archive']
                    ]
                    print(f'{plural_word(len(df_group), "file")} with `{group_name}` group.', file=f)
                    print(f'', file=f)
                    print(df_group.to_markdown(index=False), file=f)
                    print(f'', file=f)

            upload_directory_as_directory(
                repo_id=dst_repo,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add package {pack_id!r}, with {plural_word(new_row_count, "file")}'
            )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        src_repo=os.environ['REMOTE_REPOSITORY_ORD'],
        dst_repo=os.environ['REMOTE_REPOSITORY_ORD_IDX']
    )
