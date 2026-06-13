"""TTL sweep removes old result dirs, keeps fresh ones. torch-free."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sweep_expired(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("RESULT_TTL_SECONDS", "100")
    # import after env is set so module constants pick it up
    import importlib

    import worker.jobs as jobs

    importlib.reload(jobs)

    old = tmp_path / "old_job"
    new = tmp_path / "new_job"
    old.mkdir()
    new.mkdir()
    # backdate the old dir well beyond the TTL
    past = time.time() - 10_000
    os.utime(old, (past, past))

    removed = jobs.sweep_expired_results()
    assert removed == 1
    assert not old.exists()
    assert new.exists()
