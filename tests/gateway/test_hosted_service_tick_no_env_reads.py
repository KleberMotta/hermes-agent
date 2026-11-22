"""The hosted-service watcher ticks every second: an installed service must cost no profile ``.env`` read."""
from types import SimpleNamespace

import pytest

TICKS = 25


def _runner(home):
    return SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), session_control_server=None)


@pytest.mark.asyncio
async def test_installed_hosted_service_reads_no_env_per_tick(tmp_path, monkeypatch):
    import gateway.run
    from gateway.session_hosted_service import _ensure_hosted_service
    scopes = []
    real = gateway.run._profile_runtime_scope

    def counting_scope(*args, **kwargs):
        scopes.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(gateway.run, '_profile_runtime_scope', counting_scope)
    (tmp_path / '.env').write_text('SOME_KEY=value\n')
    service = SimpleNamespace(_transport_installed=True)
    authority = SimpleNamespace(profile_id=str(tmp_path), runner=_runner(tmp_path), hosted_room_service=service)
    for _ in range(TICKS):
        await _ensure_hosted_service(authority.runner, authority)
    assert scopes == []


@pytest.mark.asyncio
async def test_hosted_service_still_installs_its_transport_once_inside_the_owner_scope(tmp_path, monkeypatch):
    """The early return must not skip the first install: a service without its transport is prepared in scope."""
    import gateway.run
    import gateway.session_hosted_service as hosted
    entered, installs = [], []
    real = gateway.run._profile_runtime_scope

    def counting_scope(*args, **kwargs):
        entered.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(gateway.run, '_profile_runtime_scope', counting_scope)
    monkeypatch.setattr('gateway.session_hosted_transport.install_hosted_transport',
                        lambda *args, **kwargs: installs.append(args))
    service = SimpleNamespace(_transport_installed=False, attest=lambda *a, **k: None)
    authority = SimpleNamespace(profile_id=str(tmp_path), runner=_runner(tmp_path), hosted_room_service=service)
    for _ in range(3):
        await hosted._ensure_hosted_service(authority.runner, authority)
    assert len(installs) == 1 and service._transport_installed is True
    assert len(entered) == 1
