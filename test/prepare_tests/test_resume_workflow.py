import datetime as dt
import json
import os
import stat
import subprocess

import pytest
import yaml

WORKFLOW = os.path.join(os.path.dirname(__file__), '..', '..', '.github', 'workflows', 'newest_resume.yml')

FAKE_GH = '''#!/bin/bash
# stands in for the gh cli: answers from the environment and records what would have been started
case "$*" in
  *"--status completed"*) [ -n "$LAST_JSON" ] && echo "$LAST_JSON" ;;
  *"workflow run"*) echo "DISPATCHED" ;;
  *"run list"*)
    read -r first rest < "$SEQ_FILE"
    echo "${first:-0}"
    echo "$rest" > "$SEQ_FILE"
    ;;
esac
'''
FAKE_SLEEP = '#!/bin/bash\necho "SLEPT $1"\n'


def resume_script() -> str:
    steps = yaml.safe_load(open(WORKFLOW))['jobs']['resume']['steps']
    return next(step['run'] for step in steps if step['name'] == 'Resume')


def last_run(conclusion, lasted, title='Newest'):
    start = dt.datetime(2026, 10, 7, 10, 0, 0)
    return json.dumps({'databaseId': 111, 'conclusion': conclusion, 'displayTitle': title,
                       'startedAt': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                       'updatedAt': (start + dt.timedelta(seconds=lasted)).strftime('%Y-%m-%dT%H:%M:%SZ')})


def run(tmp_path, last=None, active='0'):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in (('gh', FAKE_GH), ('sleep', FAKE_SLEEP)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    (tmp_path / 'seq').write_text(active + '\n')
    script = tmp_path / 'resume.sh'
    script.write_text(resume_script())
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'REPO': 'a/b', 'GH_TOKEN': 'x',
           'LAST_JSON': last or '', 'ACTIVE_SEQ': active, 'SEQ_FILE': str(tmp_path / 'seq')}
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout


@pytest.mark.unittest
class TestResumeDecisions:
    def test_nothing_to_do_while_a_run_is_going(self, tmp_path):
        code, out = run(tmp_path, last_run('success', 20000), active='1')
        assert code == 0 and 'DISPATCHED' not in out and 'nothing to do' in out

    def test_a_clean_finish_without_a_successor_starts_one(self, tmp_path):
        code, out = run(tmp_path, last_run('success', 19800))
        assert code == 0 and 'DISPATCHED' in out

    def test_no_history_at_all_starts_one(self, tmp_path):
        code, out = run(tmp_path, None)
        assert code == 0 and 'DISPATCHED' in out

    def test_an_early_cancel_is_a_deliberate_stop(self, tmp_path):
        code, out = run(tmp_path, last_run('cancelled', 600))
        assert code == 0 and 'DISPATCHED' not in out and 'staying stopped' in out

    def test_a_cancel_after_the_time_limit_is_not_a_stop(self, tmp_path):
        # a job killed at the 6 hour limit may be reported as cancelled
        code, out = run(tmp_path, last_run('cancelled', 21500))
        assert code == 0 and 'DISPATCHED' in out

    @pytest.mark.parametrize('conclusion', ['failure', 'timed_out', 'startup_failure', 'neutral'])
    def test_a_late_failure_starts_a_new_run_at_once(self, tmp_path, conclusion):
        code, out = run(tmp_path, last_run(conclusion, 5000))
        assert code == 0 and 'DISPATCHED' in out and 'SLEPT' not in out

    def test_a_crash_right_after_starting_waits_and_then_starts_again(self, tmp_path):
        code, out = run(tmp_path, last_run('failure', 120), active='0 0')
        assert code == 0 and 'SLEPT 1200' in out and out.index('SLEPT 1200') < out.index('DISPATCHED')

    def test_a_run_that_appeared_during_the_wait_is_respected(self, tmp_path):
        code, out = run(tmp_path, last_run('failure', 120), active='0 1')
        assert code == 0 and 'SLEPT 1200' in out and 'DISPATCHED' not in out

    @pytest.mark.parametrize('title', ['Newest (once)', 'Newest (no chain)', 'Newest (once) (no chain)'])
    def test_debugging_runs_are_left_alone(self, tmp_path, title):
        code, out = run(tmp_path, last_run('success', 100, title))
        assert code == 0 and 'DISPATCHED' not in out and 'debugging run' in out


@pytest.mark.unittest
def test_both_triggers_are_present():
    triggers = yaml.safe_load(open(WORKFLOW))[True]
    assert triggers['workflow_run']['types'] == ['completed'] and triggers['schedule']
