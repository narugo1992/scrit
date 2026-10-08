import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

# The newest crawler only runs on Linux; these tests drive its bash workflow or POSIX signals.
NEEDS_LINUX = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='crawler runs on Linux only')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

CHILD = textwrap.dedent('''
    import os, signal, sys, time
    os.environ.setdefault('REMOTE_REPOSITORY', 'a/b')
    from test.prepare.runner import Runner, RunConfig
    from test.prepare_tests.test_runner import MemStore, FakeSkeb, FakeClient

    store = MemStore()
    runner = Runner(store, FakeSkeb([], {}), None,
                    RunConfig(budget_seconds=float(sys.argv[1]), hard_extra=float(sys.argv[2]), hard_deadline=True),
                    use_lease=True)
    store.acquire_lease = lambda holder, **kw: setattr(store, '_holding', holder)
    store.keep_lease = lambda holder: True
    released = []
    store.release_lease = lambda holder: (released.append(holder), print('RELEASED', flush=True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # what the CLI installs

    def stuck(*a, **k):
        print('READY', flush=True)
        time.sleep(60)          # a blocking call that a signal has to get through

    runner.cycle = stuck
    runner.run()
''')


def spawn(tmp_path, budget, extra):
    script = tmp_path / 'child.py'
    script.write_text(CHILD)
    return subprocess.Popen([sys.executable, str(script), str(budget), str(extra)], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH': ROOT}, stdout=subprocess.PIPE, text=True)


def wait_ready(proc):
    for line in proc.stdout:
        if line.strip() == 'READY':
            return
    raise AssertionError('the child never got ready')


@NEEDS_LINUX
@pytest.mark.unittest
class TestRealProcessEndings:
    def test_sigterm_ends_the_process_and_releases_the_lease(self, tmp_path):
        proc = spawn(tmp_path, 3600, 600)
        wait_ready(proc)
        proc.send_signal(signal.SIGTERM)
        rest = proc.stdout.read()
        assert proc.wait(timeout=20) == 143 and 'RELEASED' in rest

    def test_the_hard_deadline_ends_a_run_stuck_in_a_blocking_call(self, tmp_path):
        started = time.time()
        proc = spawn(tmp_path, 1, 1)  # budget 1s plus 1s of grace
        wait_ready(proc)
        rest = proc.stdout.read()
        code = proc.wait(timeout=30)
        assert code == 5 and 'RELEASED' in rest
        assert time.time() - started < 15  # nowhere near the 60 s the stuck call wanted
