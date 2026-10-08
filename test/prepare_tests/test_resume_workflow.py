import datetime as dt
import json
import os
import stat
import subprocess

import sys

import pytest
import yaml

# The newest crawler only runs on Linux; these tests drive its bash workflow or POSIX signals.
NEEDS_LINUX = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='crawler runs on Linux only')

WORKFLOW = os.path.join(os.path.dirname(__file__), '..', '..', '.github', 'workflows', 'newest_resume.yml')

# stands in for the gh cli: answers the run list from files (first call, later calls) and records a start
FAKE_GH = '''#!/bin/bash
case "$*" in
  *"workflow run"*) echo "DISPATCHED" ;;
  *"--status"*) echo "the status filter must not be used: it returns stale runs" >&2; exit 9 ;;
  *"run list"*)
    n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0); echo $((n + 1)) > "$COUNT_FILE"
    if [ "$n" -eq 0 ]; then cat "$RUNS_FILE"; else cat "$RUNS_AFTER_FILE"; fi
    ;;
esac
'''
FAKE_SLEEP = '#!/bin/bash\necho "SLEPT $1"\n'
START = dt.datetime(2026, 10, 7, 10, 0, 0)


def stamp(delta=0):
    return (START + dt.timedelta(seconds=delta)).strftime('%Y-%m-%dT%H:%M:%SZ')


def entry(run_id, status='completed', conclusion='success', lasted=19800, title='Newest'):
    return {'databaseId': run_id, 'status': status, 'conclusion': conclusion, 'displayTitle': title,
            'startedAt': stamp(), 'updatedAt': stamp(lasted)}


def resume_script() -> str:
    steps = yaml.safe_load(open(WORKFLOW))['jobs']['resume']['steps']
    return next(step['run'] for step in steps if step['name'] == 'Resume')


def run(tmp_path, runs, after=None, finished=None):
    """``finished``: the run that triggered the workflow (conclusion, lasted, title); None means the cron."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in (('gh', FAKE_GH), ('sleep', FAKE_SLEEP)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    (tmp_path / 'runs.json').write_text(json.dumps(runs))
    (tmp_path / 'after.json').write_text(json.dumps(after if after is not None else runs))
    script = tmp_path / 'resume.sh'
    script.write_text(resume_script())
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'REPO': 'a/b', 'GH_TOKEN': 'x',
           'RUNS_FILE': str(tmp_path / 'runs.json'), 'RUNS_AFTER_FILE': str(tmp_path / 'after.json'),
           'COUNT_FILE': str(tmp_path / 'count'), 'FINISHED': '', 'F_CONCLUSION': '', 'F_TITLE': '',
           'F_STARTED': '', 'F_UPDATED': ''}
    if finished:
        conclusion, lasted, title = finished
        env.update(FINISHED='500', F_CONCLUSION=conclusion, F_TITLE=title, F_STARTED=stamp(), F_UPDATED=stamp(lasted))
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout


def stale_entry(lasted=19800):
    """The finished run as the lagging list still shows it: going."""
    return entry(500, status='in_progress', conclusion=None, lasted=lasted)


@NEEDS_LINUX
@pytest.mark.unittest
class TestResumeAfterAnEvent:
    def test_a_finished_run_that_the_stale_list_still_shows_as_going_does_not_block_the_restart(self, tmp_path):
        # the live failure: the event arrived before the run list stopped calling the finished run active
        code, out = run(tmp_path, [stale_entry()], finished=('failure', 20000, 'Newest'))
        assert code == 0 and 'DISPATCHED' in out and 'nothing to do' not in out

    def test_another_active_run_still_prevents_a_start(self, tmp_path):
        code, out = run(tmp_path, [stale_entry(), entry(600, status='in_progress', conclusion=None)],
                        finished=('success', 19800, 'Newest'))
        assert code == 0 and 'DISPATCHED' not in out and 'nothing to do' in out

    def test_a_clean_finish_without_a_successor_starts_one(self, tmp_path):
        code, out = run(tmp_path, [stale_entry()], finished=('success', 19800, 'Newest'))
        assert 'DISPATCHED' in out and 'SLEPT 30' in out  # it lets the list settle first

    def test_an_early_cancel_is_a_deliberate_stop(self, tmp_path):
        code, out = run(tmp_path, [stale_entry(600)], finished=('cancelled', 600, 'Newest'))
        assert code == 0 and 'DISPATCHED' not in out and 'staying stopped' in out

    def test_a_cancel_after_the_time_limit_is_not_a_stop(self, tmp_path):
        # a job killed at the 6 hour limit may be reported as cancelled
        code, out = run(tmp_path, [stale_entry(21500)], finished=('cancelled', 21500, 'Newest'))
        assert 'DISPATCHED' in out

    @pytest.mark.parametrize('conclusion', ['failure', 'timed_out', 'startup_failure', 'neutral'])
    def test_a_late_failure_starts_a_new_run_at_once(self, tmp_path, conclusion):
        code, out = run(tmp_path, [stale_entry(5000)], finished=(conclusion, 5000, 'Newest'))
        assert 'DISPATCHED' in out and 'SLEPT 1200' not in out

    def test_a_crash_right_after_starting_waits_and_then_starts_again(self, tmp_path):
        code, out = run(tmp_path, [stale_entry(120)], finished=('failure', 120, 'Newest'))
        assert 'SLEPT 1200' in out and out.index('SLEPT 1200') < out.index('DISPATCHED')

    def test_a_run_that_appeared_during_the_wait_is_respected(self, tmp_path):
        code, out = run(tmp_path, [stale_entry(120)], after=[entry(700, status='queued', conclusion=None)],
                        finished=('failure', 120, 'Newest'))
        assert 'SLEPT 1200' in out and 'DISPATCHED' not in out

    @pytest.mark.parametrize('title', ['Newest (once)', 'Newest (no chain)', 'Newest (once) (no chain)'])
    def test_debugging_runs_are_left_alone(self, tmp_path, title):
        code, out = run(tmp_path, [stale_entry(100)], finished=('success', 100, title))
        assert 'DISPATCHED' not in out and 'debugging run' in out


@NEEDS_LINUX
@pytest.mark.unittest
class TestResumeFromTheCron:
    def test_nothing_is_started_while_a_run_is_going(self, tmp_path):
        code, out = run(tmp_path, [entry(900, status='in_progress', conclusion=None), entry(800)])
        assert 'DISPATCHED' not in out and 'SLEPT' not in out

    def test_the_latest_finished_run_decides_not_an_old_one(self, tmp_path):
        # the cancelled run is newer than the successful one, as in the live list
        code, out = run(tmp_path, [entry(900, conclusion='cancelled', lasted=100), entry(100, conclusion='success')])
        assert 'DISPATCHED' not in out and 'staying stopped' in out

    def test_no_history_at_all_starts_one(self, tmp_path):
        code, out = run(tmp_path, [])
        assert 'DISPATCHED' in out

    def test_a_dead_crawler_is_revived(self, tmp_path):
        code, out = run(tmp_path, [entry(900, conclusion='failure', lasted=21000)])
        assert 'DISPATCHED' in out


@pytest.mark.unittest
def test_both_triggers_are_present():
    triggers = yaml.safe_load(open(WORKFLOW))[True]
    assert triggers['workflow_run']['types'] == ['completed'] and triggers['schedule']
