"""articulated — furniture that opens: Infinigen-Sim assets sized to a scanned box.

`infinigen` decides which scanned objects have an Infinigen factory (doors, windows, cabinets,
drawer chests, dishwashers, fridges, ovens), builds them — in a local Infinigen env
(``$LR_INFINIGEN_PYTHON``) or on Modal (``litereality-infinigen``) — caches the exports, and turns
one into MuJoCo bodies with our masses, joint dynamics and photo colours. `infinigen_worker` is
the script that runs INSIDE the Infinigen env; nothing here imports Blender.
"""

from __future__ import annotations

from litereality_agent.models.articulated.infinigen import (
    ARTICULATED_DIR,
    DEFAULT_SEED,
    FACTORIES,
    INFINIGEN_COMMIT,
    MODAL_APP,
    MODAL_FUNCTION,
    PLINTH,
    Asset,
    AssetCache,
    Built,
    FakeRunner,
    LocalRunner,
    ModalRunner,
    Overrides,
    Placement,
    Runner,
    RunResult,
    Spec,
    build,
    clean,
    door_spec,
    fake_export,
    generate,
    load_asset,
    main_bounds,
    runner_from_env,
    spec_for_object,
    standalone_xml,
    tint_texture,
    unpack_zip,
    window_spec,
    write_files,
    zip_folder,
)

__all__ = ["ARTICULATED_DIR", "DEFAULT_SEED", "FACTORIES", "INFINIGEN_COMMIT", "MODAL_APP",
           "MODAL_FUNCTION", "PLINTH", "Asset", "AssetCache", "Built", "FakeRunner",
           "LocalRunner", "ModalRunner", "Overrides", "Placement", "RunResult", "Runner", "Spec",
           "build", "clean", "door_spec", "fake_export", "generate", "load_asset", "main_bounds",
           "runner_from_env", "spec_for_object", "standalone_xml", "tint_texture", "unpack_zip",
           "window_spec", "write_files", "zip_folder"]
