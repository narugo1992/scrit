import os

import requests


def github_run_finished(run_id: str) -> bool:
    """Whether the GitHub Actions run with this id has ended; False when unknown, so nothing is taken over by mistake."""
    repository = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not (repository and token and str(run_id).isdigit()):
        return False
    try:
        resp = requests.get(f'https://api.github.com/repos/{repository}/actions/runs/{run_id}',
                            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                            timeout=20)
    except requests.RequestException:
        return False
    return resp.status_code == 200 and resp.json().get('status') == 'completed'
