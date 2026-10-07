import sys

import click
from ditk import logging

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
def cli():
    pass  # pragma: no cover


@cli.command('newest', context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('--budget-minutes', type=float, default=330.0, show_default=True,
              help='Stop polling when this much time has passed (the run ends cleanly).')
@click.option('--poll-seconds', type=float, default=600.0, show_default=True,
              help='Pause between two polls of the newest-works listing.')
@click.option('--bootstrap', type=int, default=400, show_default=True,
              help='How many posts to take when the repository has no cursor yet.')
@click.option('--max-skeb-requests', type=int, default=4500, show_default=True,
              help='Hard cap of requests sent to skeb.jp by one run.')
@click.option('--skeb-interval', type=float, default=2.5, show_default=True,
              help='Minimum seconds between two requests to skeb.jp.')
@click.option('--once', is_flag=True, help='Run a single poll and exit.')
def newest(budget_minutes, poll_seconds, bootstrap, max_skeb_requests, skeb_interval, once):
    logging.try_init_root(logging.INFO)
    from pyskeb.client.client import SkebClient
    from .base import _REPOSITORY, hf_client, _ensure_repository
    from .http import Fetcher
    from .runner import Runner, RunConfig
    from .store import Store

    _ensure_repository()
    runner = Runner(
        store=Store(hf_client, _REPOSITORY),
        skeb=SkebClient(min_interval=skeb_interval),
        fx=Fetcher(),
        config=RunConfig(budget_seconds=budget_minutes * 60, poll_interval=poll_seconds, bootstrap=bootstrap,
                         max_skeb_requests=max_skeb_requests, once=once),
    )
    runner.run()
    print(runner.summary(), flush=True)
    sys.exit(1 if runner.bugs else 0)


@cli.command('pack', context_settings={**GLOBAL_CONTEXT_SETTINGS})
def pack():
    logging.try_init_root(logging.INFO)
    from .repack import repack_all
    repack_all()


@cli.command('artists', context_settings={**GLOBAL_CONTEXT_SETTINGS})
def artists():
    logging.try_init_root(logging.DEBUG)
    from .artists_idx import push_artists_sqlite
    push_artists_sqlite()


if __name__ == '__main__':
    cli()
