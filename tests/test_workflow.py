"""Safety gates of the convenience workflow, without expensive training."""
import json
from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location('study', Path(__file__).parents[1] / 'study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def valid_config():
    return {
        'defender_seeds':[1], 'attacker_seeds':[2],
        'defender_updates':4, 'attacker_updates':4,
        'batch_size':8, 'validation_size':16, 'test_size':32,
        'validate_every':2, 'learning_rate':.0008, 'threads':1,
        'defender_seconds':30, 'attacker_seconds':30,
        'baseline_channels':[2,4,4,1], 'wider_channels':[4,8,8,1],
    }


def test_incomplete_budget_rejected(tmp_path):
    cfg = {'defender_seeds':[1], 'defender_updates':4}
    parent = tmp_path / 'defender_1'; parent.mkdir()
    for state, updates in [('time_limit',2), ('complete',3), ('failed',4)]:
        (parent/'status.json').write_text(json.dumps({'status':state,'updates':updates}))
        with pytest.raises(RuntimeError): study.require_complete(tmp_path,cfg)


def test_failed_gate_never_evaluates(tmp_path, monkeypatch):
    config = tmp_path/'input.json'
    config.write_text(json.dumps(valid_config()))
    calls = []
    def fake_execute(args, log=None):
        calls.append(args)
        if 'defender' in args:
            parent=tmp_path/'run/defender_1'; parent.mkdir()
            (parent/'status.json').write_text('{"status":"time_limit","updates":2}')
    monkeypatch.setattr(study,'execute',fake_execute)
    monkeypatch.setattr(study,'PYTHON',Path(__file__))
    with pytest.raises(RuntimeError): study.workflow(tmp_path/'run',config)
    assert not any('attack' in args or 'evaluate' in args for args in calls)
    assert json.loads((tmp_path/'run/workflow_status.json').read_text())['status']=='failed'
    assert (tmp_path/'run/workflow_error.txt').exists()
    with pytest.raises(FileExistsError): study.workflow(tmp_path/'run',config)


def test_view_does_not_train_or_evaluate(tmp_path, monkeypatch):
    (tmp_path/'workflow_status.json').write_text('{"status":"complete"}')
    calls=[]
    monkeypatch.setattr(study,'execute',lambda args: calls.append(args))
    study.view(tmp_path)
    assert [args[3] for args in calls]==['plot','demo']
