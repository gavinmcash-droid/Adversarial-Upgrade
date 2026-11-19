"""Convenience CLI; scientific stages remain implemented in anc.run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import venv

from anc.config import validate_config

PROJECT = Path(__file__).resolve().parent
PYTHON = PROJECT / '.venv/bin/python'


def execute(args, log=None):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    env = dict(os.environ)
    env['MPLCONFIGDIR'] = str(PROJECT / '.cache/matplotlib')
    if log is None:
        subprocess.run(list(map(str, args)), cwd=PROJECT, env=env, check=True)
    else:
        print(f'  Output: {log}', flush=True)
        with log.open('w') as out:
            subprocess.run(list(map(str, args)), cwd=PROJECT, env=env,
                           stdout=out, stderr=subprocess.STDOUT, check=True)


def setup():
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=False).create(PROJECT / '.venv')
    probe = subprocess.run([str(PYTHON), '-m', 'pip', '--version'], capture_output=True)
    if probe.returncode:
        # Same local-only bootstrap as README; never installs into system Python.
        with tempfile.TemporaryDirectory(prefix='anc-setup-') as directory:
            bootstrap = Path(directory) / 'get-pip.py'
            urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', bootstrap)
            execute([PYTHON, bootstrap])
    execute([PYTHON, '-m', 'pip', 'install', '--extra-index-url',
             'https://download.pytorch.org/whl/cpu', '-r', PROJECT / 'requirements.lock.txt'])


def require_complete(root, cfg, attackers=False):
    for seed in cfg['defender_seeds']:
        parent = root / f'defender_{seed}'
        paths = [parent] if not attackers else [
            parent / f'attack_{name}_{init}'
            for init in cfg['attacker_seeds'] for name in ('baseline', 'wider')]
        expected = cfg['attacker_updates' if attackers else 'defender_updates']
        for path in paths:
            status = json.loads((path / 'status.json').read_text())
            if status.get('status') != 'complete' or status.get('updates') != expected:
                raise RuntimeError(f'Budget incomplete at {path}; preserved without final evaluation')


def workflow(root, config):
    if not PYTHON.exists():
        raise RuntimeError('Run python3 study.py setup first')
    cfg = validate_config(json.loads(config.read_text()))
    root.mkdir(parents=True, exist_ok=False)
    saved_config = root / 'config.json'
    saved_config.write_text(json.dumps(cfg, indent=2) + '\n')
    # Preserve the wrapper as well as the per-run anc source snapshots.
    (root / 'study.py').write_bytes(Path(__file__).read_bytes())
    status_path = root / 'workflow_status.json'
    stage = 'verification'
    def status(state):
        status_path.write_text(json.dumps({'status':state, 'stage':stage}, indent=2)+'\n')
    try:
        status('running')
        execute([PYTHON, '-m', 'pytest', '-q'], root / 'verification.log')
        for stage in ('defender', 'attack', 'evaluate'):
            status('running')
            print(f'Running {stage} stage', flush=True)
            execute([PYTHON, '-m', 'anc.run', stage, '--config', saved_config,
                     '--root', root], root / f'{stage}.log')
            if stage in ('defender', 'attack'):
                require_complete(root, cfg, attackers=stage == 'attack')
        status('complete')
    except BaseException:
        status('failed')
        (root / 'workflow_error.txt').write_text(traceback.format_exc())
        raise
    print(f'Complete. Results: {root}\nNext: python3 study.py view --root {root}')


def view(root):
    cfg = root / 'config.json'
    status = json.loads((root / 'workflow_status.json').read_text())
    if status['status'] != 'complete':
        raise RuntimeError('Workflow is incomplete; inspect its logs and status first')
    # These commands read saved results/checkpoints; never optimize or evaluate again.
    for stage in ('plot', 'demo'):
        execute([PYTHON, '-m', 'anc.run', stage, '--config', cfg, '--root', root])
    for parent in sorted(root.glob('defender_*')):
        print(f'\nFinal metrics: {parent / "test_results.json"}')
        print((parent / 'test_results.json').read_text())
        print(f'Open plot: {parent / "validation.png"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup', help='Install locked dependencies into local .venv')
    run = sub.add_parser('run', help='Verify, train defenders and fresh Eves, evaluate once')
    source = run.add_mutually_exclusive_group()
    source.add_argument('--profile', choices=['smoke','pilot','full'], default='pilot')
    source.add_argument('--config', type=Path, help='Custom predeclared configuration')
    run.add_argument('--root', required=True, type=Path, help='New output directory')
    show = sub.add_parser('view', help='Plot saved histories and display predictions/metrics')
    show.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'setup':
        setup()
    elif args.command == 'run':
        config = args.config or PROJECT / 'configs' / f'{args.profile}.json'
        workflow(args.root.resolve(), config.resolve())
    else:
        view(args.root.resolve())


if __name__ == '__main__':
    main()
