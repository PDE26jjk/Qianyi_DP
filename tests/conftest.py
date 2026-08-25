"""Shared pytest configuration and fixtures for the Qianyi_DP test suite."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.report import write_results
from harness.output_redirect import RedirectRouter

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_NAME = "Qianyi_DP"
PYD_PATTERN = "Qianyi_DP.cp*-*.pyd"
ABI_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

BUILD_HINT = (
    "Build Qianyi_DP with CMake (see AGENTS.md / LOCAL_DEV.md), then either set "
    "the QYDP_PYD environment variable to the .pyd file or run pytest from the "
    "repository root so the build output directories are scanned."
)

FINGERPRINT_HINT = (
    "The Qianyi_DP module is missing the expected fingerprint (the `test` "
    "submodule and a readable `__version__`). It is likely a stale or partial "
    "build; rebuild the module with CMake and retry."
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gif",
        action="store_true",
        default=False,
        help="Generate GIF artifacts for visual simulation tests.",
    )


# @pytest.fixture(scope="session")
# def artifact_root() -> Path:
#     """Default artifact root (gitignored ``tests/artifacts``)."""
#     return REPO_ROOT / "tests" / "artifacts"
#

@pytest.fixture(scope="session")
def capture(pytestconfig):
    """Persistent fd-level output capture with per-case log switching.

    Use ``capture.begin_case(log_path)`` before and ``capture.end_case()``
    after each simulation case so C-level prints land in the per-case log.
    """
    router = RedirectRouter(verbose=pytestconfig.getoption("verbose") > 0)
    yield router
    router.stop()


@pytest.fixture
def record_failure(request):
    """Register artifact/failure metadata on the node for the report hook.

    Tests call ``record_failure(artifact_dir, test_id, **failure_data)`` before
    their assertions; if an assertion then fails, the hook writes a
    ``results.json`` with status ``failed`` plus the per-frame stats and prints
    a "view <artifact path>" hint.
    """

    def _register(artifact_dir_path: Path, test_id: str, **failure_data) -> None:
        request.node._qydp_artifact_dir = Path(artifact_dir_path)
        request.node._qydp_test_id = test_id
        request.node._qydp_failure_data = failure_data

    return _register


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Write failure results.json and print the artifact hint on sim failures."""
    outcome = yield
    report_obj = outcome.get_result()
    if report_obj.when != "call" or not report_obj.failed:
        return
    artifact = getattr(item, "_qydp_artifact_dir", None)
    test_id = getattr(item, "_qydp_test_id", None)
    if artifact is None or test_id is None:
        return
    failure: dict = {
        "message": str(call.excinfo.value) if call.excinfo else str(report_obj.longrepr)
    }
    extra = getattr(item, "_qydp_failure_data", None)
    if extra:
        failure.update(extra)
    write_results(
        test_id,
        status="failed",
        artifacts={"dir": str(artifact)},
        failure=failure,
    )
    print(f"\nview artifact dir: {artifact}")


def _candidate_pyd_paths() -> list[Path]:
    """Return candidate Qianyi_DP .pyd files for the running interpreter."""
    candidates: list[Path] = []

    env_value = os.environ.get("QYDP_PYD")
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_file():
            return [env_path]
        if env_path.is_dir():
            candidates.extend(env_path.glob(PYD_PATTERN))
        else:
            raise RuntimeError(
                f"QYDP_PYD points to a path that does not exist: {env_value}\n"
                f"{BUILD_HINT}"
            )

    for scan_root in (REPO_ROOT / "build", REPO_ROOT / "dist"):
        if scan_root.is_dir():
            candidates.extend(scan_root.rglob(PYD_PATTERN))

    # Only modules built for the running interpreter can be imported.
    abi_candidates = [c for c in candidates if ABI_TAG in c.name]
    unique: dict[Path, Path] = {}
    for candidate in abi_candidates:
        unique.setdefault(candidate.resolve(), candidate)
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_module(pyd_path: Path):
    """Import Qianyi_DP from an explicit .pyd location."""
    module_dir = pyd_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    return importlib.import_module(MODULE_NAME)


def _check_fingerprint(module) -> None:
    """Verify the module is a usable Qianyi_DP build (test submodule + version)."""
    if not hasattr(module, "test"):
        raise RuntimeError(f"{FINGERPRINT_HINT} (missing 'test' submodule)")
    if not hasattr(module, "__version__"):
        raise RuntimeError(f"{FINGERPRINT_HINT} (missing '__version__')")


def _plain_import_available() -> bool:
    """Probe (in a subprocess) whether a plain ``import Qianyi_DP`` is safe.

    A stale or broken installed .pyd can hard-crash the process (e.g. missing
    adjacent runtime DLLs); probing in a subprocess keeps the test session
    alive so the fixture can skip with build instructions instead.
    """
    probe = "import importlib; importlib.import_module('Qianyi_DP'); print('OK')"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=60,
        )
    except Exception:
        return False
    return result.returncode == 0 and b"OK" in result.stdout


def _ensure_shader_assets(module) -> None:
    """Copy the rasterizer shaders next to the module if missing.

    The CUDA extension resolves its Vulkan shader directory to the module's
    own directory (``g_module_dir``); without ``assets/vert.spv`` and
    ``assets/frag.spv`` next to the .pyd, ``sample_points`` fails with
    "Shader not found". The repo copies live under ``src/graphics/assets``.
    """
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return
    module_dir = Path(module_file).resolve().parent
    assets_dir = module_dir / "assets"
    if (assets_dir / "vert.spv").exists() and (assets_dir / "frag.spv").exists():
        return
    source_assets = REPO_ROOT / "src" / "graphics" / "assets"
    if not (source_assets / "vert.spv").exists() or not (source_assets / "frag.spv").exists():
        return
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_assets / "vert.spv", assets_dir / "vert.spv")
    shutil.copy2(source_assets / "frag.spv", assets_dir / "frag.spv")


def _resolve_qydp():
    """Resolve the module: QYDP_PYD env var -> ABI-tagged build scan -> import."""
    candidates = _candidate_pyd_paths()
    if candidates:
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                module = _load_module(candidate)
                _check_fingerprint(module)
                _ensure_shader_assets(module)
                return module
            except Exception as exc:  # noqa: BLE001 - surface the last failure
                last_error = exc
        raise RuntimeError(
            "Qianyi_DP could not be loaded from: "
            f"{[str(c) for c in candidates]}\n"
            f"Last error: {last_error}\n{BUILD_HINT}"
        )

    if not _plain_import_available():
        return None
    try:
        module = importlib.import_module(MODULE_NAME)
        _check_fingerprint(module)
        _ensure_shader_assets(module)
        return module
    except Exception:
        return None


@pytest.fixture(scope="session")
def qydp():
    """Resolve and return the Qianyi_DP extension module.

    Resolution order: QYDP_PYD env var -> scan the repo's build output
    directories for the current interpreter's ABI tag -> plain import.
    Tests that depend on the module are skipped with build instructions when
    it cannot be found; a found-but-broken module raises instead.
    """
    module = _resolve_qydp()
    if module is None:
        pytest.skip(f"Qianyi_DP module not found. {BUILD_HINT}")
    return module
