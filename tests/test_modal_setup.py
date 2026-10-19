from __future__ import annotations

import ast
from pathlib import Path

import pytest

from litereality_agent.runtimes import setup

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "modal"
# What a developer's shell may export (deploy/modal/README.md tells hand-deployers to), and what
# the settings and ModalClient read. Cleared for every test so none of them depends on the shell.
MODAL_ENV = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_PROFILE",
    "MODAL_ENVIRONMENT",
    "MODAL_TRELLIS_APP",
    "MODAL_TRELLIS_FUNCTION",
    "MODAL_DINO_APP",
    "MODAL_DINO_FUNCTION",
    "MODAL_IMAGE_APP",
    "MODAL_IMAGE_FUNCTION",
)


@pytest.fixture(autouse=True)
def no_modal_shell_env(monkeypatch):
    for key in MODAL_ENV:
        monkeypatch.delenv(key, raising=False)


def test_record_profile_creates_env_file(tmp_path):
    env = tmp_path / ".env"

    assert setup.record_profile(env, "team-a") is True
    assert env.read_text(encoding="utf-8") == "MODAL_PROFILE=team-a\n"


def test_record_profile_appends_without_clobbering(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    assert setup.record_profile(env, "team-a") is True

    body = env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test" in body
    assert "MODAL_PROFILE=team-a" in body


def test_record_profile_rewrites_in_place_on_switch(tmp_path):
    """Switching workspaces must not leave two conflicting assignments behind."""
    env = tmp_path / ".env"
    env.write_text("MODAL_PROFILE=old\nBLENDER_PATH=/b\n", encoding="utf-8")

    assert setup.record_profile(env, "new") is True

    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines == ["MODAL_PROFILE=new", "BLENDER_PATH=/b"]
    assert sum(line.startswith("MODAL_PROFILE=") for line in lines) == 1


def test_record_profile_is_a_noop_when_already_correct(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MODAL_PROFILE=team-a\n", encoding="utf-8")

    assert setup.record_profile(env, "team-a") is False


def test_deploy_reports_a_missing_checkout(tmp_path):
    """Installed-from-wheel users have no deploy/ tree; the error has to say so."""
    with pytest.raises(setup.SetupError, match="source checkout"):
        setup.deploy(tmp_path, "main", "team-a")


def test_deploy_runs_each_app_with_the_profile_pinned(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs.get("env", {}).get("MODAL_PROFILE")))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(setup, "_run", fake_run)
    setup.deploy(tmp_path, "prod", "team-a")

    assert len(calls) == len(setup.APPS) == 1, "the combined app deploys in one call"
    for command, profile in calls:
        assert command[1:4] == ["-m", "modal", "deploy"]
        assert command[4:6] == ["--env", "prod"]
        assert profile == "team-a", "each deploy must be pinned to the chosen profile"
    assert str(tmp_path / setup.APPS[0][1]) == calls[0][0][-1]


def test_deploy_raises_when_a_deploy_fails(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    monkeypatch.setattr(setup, "_run", lambda *a, **k: type("R", (), {"returncode": 1})())

    with pytest.raises(setup.SetupError, match=setup.APPS[0][0]):
        setup.deploy(tmp_path, "main", "team-a")


def module_constants(app_file: Path) -> dict[str, str]:
    """Module-level `NAME = "string"` assignments of a Modal app file.

    Every check on the deploy layout reads the files with ast rather than importing them, so it
    needs neither the Modal client nor the per-model sub-apps to be importable.
    """
    tree = ast.parse(app_file.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _deployed_app_name(app_file: Path) -> tuple[str, str]:
    """`(APP_NAME, name passed to the module-level modal.App(...))` for a Modal app file.

    The second is what `modal deploy` names the app.
    """
    tree = ast.parse(app_file.read_text(encoding="utf-8"))
    constants = module_constants(app_file)
    for node in tree.body:
        call = node.value if isinstance(node, ast.Assign) else None
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "App"
            and call.args
        ):
            name = call.args[0]
            deployed = constants[name.id] if isinstance(name, ast.Name) else name.value
            return constants["APP_NAME"], deployed
    raise AssertionError(f"no module-level `modal.App(<name>)` in {app_file}")


def test_setup_deploys_the_app_the_runtime_looks_up():
    """Regression: setup deployed `litereality-dino` and `litereality-trellis` while the runtime
    looked up one `litereality` app, so setup succeeded and every hosted call was app-not-found.
    """
    from litereality_agent.settings import LiteRealitySettings

    assert [relative for _, relative in setup.APPS] == ["deploy/modal/app.py"], (
        "a Modal deploy replaces every function on the app; deploy the combined app, whole"
    )

    app_name, deployed = _deployed_app_name(REPO / setup.APPS[0][1])
    assert deployed == app_name

    settings = LiteRealitySettings(_env_file=None)
    assert settings.modal_trellis_app == app_name
    assert settings.modal_dino_app == app_name
    assert settings.modal_image_app == app_name


def test_settings_point_at_the_deployed_functions():
    from litereality_agent.settings import LiteRealitySettings

    settings = LiteRealitySettings(_env_file=None)
    for directory, function in (
        ("trellis", settings.modal_trellis_function),
        ("dino", settings.modal_dino_function),
        ("image_edit", settings.modal_image_function),
    ):
        deployed = module_constants(DEPLOY / directory / "app.py")["FUNCTION_NAME"]
        assert function == deployed, f"{directory}: settings look up {function!r}, not {deployed!r}"


def test_combined_app_includes_every_model_app():
    """A deploy replaces every function on the app: a per-model app that deploy/modal/app.py stops
    including is DELETED by the next `litereality setup`, and every hosted call to it is then
    app-not-found — the failure this layout exists to prevent.

    The expected set is every per-model directory that is a deployable app, so a fourth one added
    and forgotten here fails too.
    """
    tree = ast.parse((DEPLOY / "app.py").read_text(encoding="utf-8"))
    aliases = {
        node.module: alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
        if alias.name == "app"
    }
    included = {
        node.value.args[0].id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "include"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "app"
        and node.value.args
        and isinstance(node.value.args[0], ast.Name)
    }
    expected = sorted(
        f"{directory.name}.app"
        for directory in DEPLOY.iterdir()
        if (directory / "app.py").is_file() and (directory / "__init__.py").is_file()
    )
    assert {"trellis.app", "dino.app", "image_edit.app"} <= set(expected)
    for module in expected:
        assert module in aliases, f"deploy/modal/app.py never imports `app` from {module}"
        assert aliases[module] in included, (
            f"deploy/modal/app.py imports {module} but never calls app.include() on it"
        )


def test_run_quotes_the_one_time_costs_before_deploying(tmp_path, monkeypatch, capsys):
    """A first deploy must never be a surprise charge: both cold costs are printed up front."""
    monkeypatch.setattr(setup, "require_client", lambda: None)
    order = []
    monkeypatch.setattr(
        setup, "deploy", lambda *a, **k: order.append(capsys.readouterr().out)
    )

    assert setup.run(tmp_path, credentials=("ak-1", "as-2")) == 0

    before_deploy = order[0]
    assert setup.TRELLIS_BUILD_ESTIMATE in before_deploy
    assert "image_edit" in before_deploy
    assert setup.IMAGE_EDIT_FIRST_CALL_ESTIMATE in before_deploy


def test_run_skips_deploy_but_still_records_the_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "require_client", lambda: None)
    monkeypatch.setattr(
        setup, "deploy", lambda *a, **k: pytest.fail("must not deploy with --skip-deploy")
    )

    assert setup.run(tmp_path, profile="team-a", skip_deploy=True) == 0
    assert "MODAL_PROFILE=team-a" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_tokens_authenticate_without_a_browser_or_profile(tmp_path, monkeypatch):
    """The whole point of the token path: no `modal setup`, no profile, nothing written back."""
    monkeypatch.setattr(setup, "require_client", lambda: None)
    monkeypatch.setattr(
        setup, "authenticate", lambda: pytest.fail("tokens must not trigger a browser login")
    )
    deployed = {}
    monkeypatch.setattr(
        setup,
        "deploy",
        lambda root, env, profile, creds: deployed.update(profile=profile, creds=creds),
    )

    assert setup.run(tmp_path, credentials=("ak-1", "as-2")) == 0
    assert deployed == {"profile": None, "creds": ("ak-1", "as-2")}
    assert not (tmp_path / ".env").exists(), "tokens are already in .env; nothing to write back"


def test_deploy_passes_tokens_to_each_subprocess(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    seen = []
    monkeypatch.setattr(
        setup,
        "_run",
        lambda command, **kw: (
            seen.append((kw["env"].get("MODAL_TOKEN_ID"), kw["env"].get("MODAL_TOKEN_SECRET"))),
            type("R", (), {"returncode": 0})(),
        )[1],
    )

    setup.deploy(tmp_path, "main", None, ("ak-1", "as-2"))

    assert seen == [("ak-1", "as-2")] * len(setup.APPS)


def test_settings_need_both_halves_of_the_token_pair():
    """Half a pair cannot authenticate, so it must not read as configured."""
    from litereality_agent.settings import LiteRealitySettings

    half = LiteRealitySettings(_env_file=None, MODAL_TOKEN_ID="ak-1")
    assert half.modal_credentials() is None
    assert half.modal_configured() is False

    whole = LiteRealitySettings(_env_file=None, MODAL_TOKEN_ID="ak-1", MODAL_TOKEN_SECRET="as-2")
    assert whole.modal_credentials() == ("ak-1", "as-2")
    assert whole.modal_configured() is True


def test_tokens_reach_subprocesses_through_as_environment():
    from litereality_agent.settings import LiteRealitySettings

    settings = LiteRealitySettings(
        _env_file=None, MODAL_TOKEN_ID="ak-1", MODAL_TOKEN_SECRET="as-2"
    )
    env = settings.as_environment()

    assert env["MODAL_TOKEN_ID"] == "ak-1"
    assert env["MODAL_TOKEN_SECRET"] == "as-2"


def test_modal_client_rejects_having_no_credentials_at_all():
    from litereality_agent.runtimes.modal import ModalClient

    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
        ModalClient("app", "fn")


def test_cli_exposes_setup():
    from litereality_agent.cli import _parser

    args = _parser().parse_args(["setup", "--skip-deploy", "--profile", "team-a"])

    assert args.profile == "team-a"
    assert args.skip_deploy is True
