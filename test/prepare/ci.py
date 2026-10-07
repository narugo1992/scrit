import os
from typing import Optional

import requests


def github_run_status(run_id: str) -> Optional[str]:
    """Status of a GitHub Actions run (queued, in_progress, completed, ...); None when it cannot be told."""
    repository = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not (repository and token and str(run_id).isdigit()):
        return None
    try:
        resp = requests.get(f'https://api.github.com/repos/{repository}/actions/runs/{run_id}',
                            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                            timeout=20)
    except requests.RequestException:
        return None
    return resp.json().get('status') if resp.status_code == 200 else None
