import json
import logging
import os
import os.path
import zipfile
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from hbutils.scale import size_to_bytes_str
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub import hf_hub_url, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError
from tqdm.auto import tqdm
from hfutils.operate import hf_repo_glob, download_file_to_file
from pyskeb.utils.download import download_file
from .base import _REPOSITORY, hf_client, _ensure_repository

hf_token = os.environ.get('HF_TOKEN')


@contextmanager
def repack_zips(max_size_limit: Optional[float] = None):
    with TemporaryDirectory() as td:
        dd_dir = os.path.join(td, 'origin')
        os.makedirs(dd_dir, exist_ok=True)

        fns = []
        current_size = 0

        # Use hf_repo_glob instead of hf_fs.glob
        zip_files = hf_repo_glob(
            repo_id=_REPOSITORY,
            pattern='unarchived/*.zip',
            repo_type='dataset',
            hf_token=hf_token
        )

        for file_item in tqdm(zip_files):
            filename = os.path.basename(file_item.path)
            if max_size_limit is not None and current_size >= max(max_size_limit * 0.95, max_size_limit - 100):
                break

            if max_size_limit is not None and current_size + file_item.size >= max_size_limit:
                continue

            try:
                with TemporaryDirectory() as ctd:
                    zip_file = os.path.join(ctd, filename)
                    download_file(
                        hf_hub_url(repo_id=_REPOSITORY, repo_type='dataset', filename=f'unarchived/{filename}'),
                        zip_file,
                        headers={'Authorization': f'Bearer {hf_token}'},
                    )
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        try:
                            zf.extractall(dd_dir)
                        except OSError as err:
                            logging.warning(repr(err))
            except:
                logging.exception(f'Error when extracting file {filename!r}, skipped.')
                continue
            else:
                current_size += file_item.size
                fns.append(filename)

        zip_file = os.path.join(td, 'package.zip')
        written = False
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for root, dirs, files in os.walk(dd_dir):
                for file in files:
                    filename = os.path.join(dd_dir, root, file)
                    relname = os.path.relpath(filename, dd_dir)
                    zf.write(filename, relname)
                    os.remove(filename)
                    written = True

        if written:
            yield zip_file, fns
        else:
            yield None, fns


def _make_records():
    # Use HfApi.file_exists instead of hf_fs.exists
    if not hf_client.file_exists(
        repo_id=_REPOSITORY,
        filename='index.json',
        repo_type='dataset'
    ):
        retval = []
        # Use hf_repo_glob instead of hf_fs.glob
        pack_files = hf_repo_glob(
            repo_id=_REPOSITORY,
            pattern='packs/*.zip',
            repo_type='dataset',
            hf_token=hf_token
        )
        for pack_item in pack_files:
            filename = os.path.basename(pack_item.path)
            size = pack_item.size
            retval.append({'filename': filename, 'size': size})
        return retval
    else:
        # Download and read the index.json file
        with TemporaryDirectory() as td:
            index_file = os.path.join(td, 'index.json')
            download_file_to_file(
                local_file=index_file,
                repo_id=_REPOSITORY,
                file_in_repo='index.json',
                repo_type='dataset',
                hf_token=hf_token
            )
            with open(index_file, 'r') as f:
                return json.load(f)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def repack_all():
    _ensure_repository()
    # Use HfApi.file_exists instead of hf_fs.exists
    if hf_client.file_exists(
        repo_id=_REPOSITORY,
        filename='archived.json',
        repo_type='dataset'
    ):
        # Download and read the archived.json file
        with TemporaryDirectory() as td:
            archived_file = os.path.join(td, 'archived.json')
            download_file_to_file(
                local_file=archived_file,
                repo_id=_REPOSITORY,
                file_in_repo='archived.json',
                repo_type='dataset',
                hf_token=hf_token
            )
            with open(archived_file, 'r') as f:
                archived_resource_ids = json.load(f)
    else:
        archived_resource_ids = []

    with repack_zips(max_size_limit=5.5 * 1024 ** 3) as (zip_file, fns):
        if zip_file is None:
            logging.info('No files to repack, skipped.')
            return

        package_name = f'pack_{_timestamp()}.zip'
        logging.info(f'Creating new pack {package_name!r} ...')
        operations = []
        operations.append(CommitOperationAdd(
            path_or_fileobj=zip_file,
            path_in_repo=f'packs/{package_name}'
        ))
        for fn in fns:
            operations.append(CommitOperationDelete(
                path_in_repo=f'unarchived/{fn}',
            ))
            archived_resource_ids.append(os.path.splitext(fn)[0])

        all_records = _make_records()
        all_records.append({'filename': package_name, 'size': os.path.getsize(zip_file)})
        all_records = sorted(all_records, key=lambda x: x['filename'], reverse=True)

        df_records = []
        for item in all_records:
            url_for_download = hf_hub_url(
                repo_id=_REPOSITORY, repo_type="dataset",
                filename=f"packs/{item['filename']}"
            )
            df_records.append({
                'Filename': item['filename'],
                'Size': size_to_bytes_str(item['size'], precision=3),
                'Link': f'[Download]({url_for_download})'
            })

        df = pd.DataFrame(df_records)

        with TemporaryDirectory() as td:
            md_file = os.path.join(td, 'README.md')
            with open(md_file, 'w') as f:
                print('---', file=f)
                print('license: other', file=f)
                print('---', file=f)
                print('', file=f)
                print(df.to_markdown(index=False), file=f)

            operations.append(CommitOperationAdd(
                path_or_fileobj=md_file,
                path_in_repo='README.md',
            ))

            index_file = os.path.join(td, 'index.json')
            with open(index_file, 'w') as f:
                json.dump(all_records, f, sort_keys=True, ensure_ascii=False, indent=4)
            operations.append(CommitOperationAdd(
                path_or_fileobj=index_file,
                path_in_repo='index.json',
            ))

            archived_json_file = os.path.join(td, 'archived.json')
            with open(archived_json_file, 'w') as f:
                json.dump(archived_resource_ids, f, indent=4, ensure_ascii=False)
            operations.append(CommitOperationAdd(
                path_or_fileobj=archived_json_file,
                path_in_repo='archived.json',
            ))

            if operations:
                while True:
                    try:
                        hf_client.create_commit(
                            repo_id=_REPOSITORY,
                            repo_type='dataset',
                            operations=operations,
                            commit_message=f'Create new package {package_name!r}.'
                        )
                    except HfHubHTTPError as err:
                        logging.exception(err)
                        logging.warning('Retry to commit ...')
                    else:
                        break
