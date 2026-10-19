"""The tool registry the live stages use — the capability tools handed to the authoring / materials /
QC agent (via ``author.build_capability_server``), and nothing else.

This used to describe an "11 primitives + 3 composites" closed set driven by a raw-API closed loop.
That driver is gone; the agent runs on the claude_agent_sdk and is handed exactly these tools as
``mcp__cap__*`` (plus Read/Edit/Write/Glob). The retired primitives and composites live under the
repo-root ``legacy/litereality_agent/realism_authoring/tools/`` for reference. See authoring/tools/README.md.
"""

from __future__ import annotations

from litereality_agent.agent.tools.check_collisions import CheckCollisionsTool
from litereality_agent.agent.tools.critic import CriticTool
from litereality_agent.agent.tools.fetch_material import FetchMaterialTool
from litereality_agent.agent.tools.grid import GridTool
from litereality_agent.agent.tools.registry import ToolRegistry
from litereality_agent.agent.tools.render import RenderTool
from litereality_agent.agent.tools.select_views import SelectViewsTool

# The capability tools (see author.CAPABILITY_TOOLS). `compile` stays a module (render imports it),
# but is not registered — the agent recompiles through render, not a standalone tool.
CAPABILITY_TOOLS = (
    FetchMaterialTool,
    RenderTool,
    CriticTool,
    SelectViewsTool,
    GridTool,
    CheckCollisionsTool,
)


def build_default_registry() -> ToolRegistry:
    return ToolRegistry([cls() for cls in CAPABILITY_TOOLS])
