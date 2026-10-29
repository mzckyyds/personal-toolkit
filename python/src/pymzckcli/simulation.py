"""虚拟世界模拟命令行工具."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from dotenv import load_dotenv
from pydantic import AwareDatetime, BaseModel, TypeAdapter, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx2


load_dotenv()


# =====================================================================================
# Helpers
# =====================================================================================
def _version() -> str:
    return f"ver_{time.time_ns() // 1_000_000}"


async def _request_hook(  # pyright: ignore[reportUnusedFunction]
    request: httpx2.Request,
) -> None:
    body = await request.aread()
    typer.echo("\n===== MODEL REQUEST =====", err=True)
    typer.echo(
        json.dumps(json.loads(body), ensure_ascii=False, indent=2),
        err=True,
    )
    typer.echo("===== END MODEL REQUEST =====\n", err=True)


# ======================================================================================
# Memory
# ======================================================================================
class BaseMemory(BaseModel):
    """基础记忆."""

    datetime: AwareDatetime
    content: str


class WorldMemory(BaseMemory):
    """世界记忆, 仅包含世界事件. content 需用角色名进行记录."""

    owner: Literal["world"]
    kind: Literal["EVENT"]


class CharacterMemory(BaseMemory):
    """角色记忆, 包含角色的观察、行动或事件. content 需带入角色视角以第一人称记录."""

    owner: str
    kind: Literal["OBSERVE", "ACTION", "EVENT"]

    @field_validator("owner")
    @classmethod
    def owner_must_not_be_world(  # noqa: D102
        cls,
        v: str,
    ) -> str:
        if v == "world":
            errmsg = "`CharacterMemory.owner` cannot be 'world'"
            raise ValueError(errmsg)
        return v


# ======================================================================================
# State
# ======================================================================================
class BaseState(BaseModel):
    """基础状态."""

    datetime: AwareDatetime
    description: str


class WorldState(BaseState):
    """世界状态."""

    owner: Literal["world"]


class CharacterState(BaseState):
    """角色状态. inventory 表示角色的物品栏, 记录角色所持物品及其数量和作用."""

    owner: str
    inventory: str

    @field_validator("owner")
    @classmethod
    def owner_must_not_be_world(  # noqa: D102
        cls,
        v: str,
    ) -> str:
        if v == "world":
            errmsg = "`CharacterState.owner` cannot be 'world'"
            raise ValueError(errmsg)
        return v


# ======================================================================================
# Observation & Action
# ======================================================================================
class Observation(BaseModel):
    """角色观察."""

    character: str
    observation: str


class Action(BaseModel):
    """角色行动."""

    character: str
    action: str


# ======================================================================================
# Executor
# ======================================================================================
@dataclass
class InitDeps:  # noqa: D101
    world: Path


@dataclass
class ObserveDeps:  # noqa: D101
    world: Path


@dataclass
class ActDeps:  # noqa: D101
    world: Path
    character: str
    observation: str


@dataclass
class JudgeDeps:  # noqa: D101
    world: Path
    observations: list[Observation]
    actions: list[Action]


executor = Agent(
    model=OpenAIChatModel(
        "deepseek-v4.1-flash",
        provider=OpenAIProvider(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
    ),
    deps_type=InitDeps | ObserveDeps | ActDeps | JudgeDeps,
)


@executor.instructions
def instructions(  # noqa: D103
    ctx: RunContext[InitDeps | ObserveDeps | ActDeps | JudgeDeps],
) -> str:
    def dump(obj: object) -> str:
        return json.dumps(
            obj, ensure_ascii=False, default=lambda o: o.model_dump(mode="json")
        )

    memory_segment = "\n<MEMORY>\n{}\n</MEMORY>\n"
    state_segment = "\n<STATE>\n{}\n</STATE>\n"
    role_segment = "\n<ROLE>\n{}\n</ROLE>\n"
    task_segment = "\n<TASK>\n{}\n</TASK>\n"

    deps = ctx.deps
    world = deps.world

    if isinstance(deps, InitDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world)))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责推进世界时间线.")
            + task_segment.format("阅读记忆与状态, 将时间线向前按需推进.")
        )

    if isinstance(deps, ObserveDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world)))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责挑选行动角色并生成观察.")
            + task_segment.format(
                "基于当前记忆与状态, 选出适合推进剧情的角色, "
                "为每个角色生成一段观察: 客观描述该角色当下能看到/听到/感受到的周边信息"
                "(不使用任何人称), character 必须是记忆和状态中存在的角色."
            )
        )

    if isinstance(deps, ActDeps):
        character = deps.character
        return (
            memory_segment.format(
                dump(Simulation.read_memories(world, owner=character))
            )
            + state_segment.format(dump(Simulation.read_states(world, owner=character)))
            + role_segment.format(f"你是角色 {character}, 以第一人称思考与行动.")
            + task_segment.format(
                f"\n<OBSERVATION>\n{deps.observation}\n</OBSERVATION>\n"
                "根据你的记忆/状态/观察, 描述你接下来的一条行动(做什么, 怎么做). "
            )
        )

    return (
        memory_segment.format(dump(Simulation.read_memories(world)))
        + state_segment.format(dump(Simulation.read_states(world)))
        + role_segment.format("你是游戏主持人(GM), 负责裁决角色行动.")
        + task_segment.format(
            f"\n<OBSERVATIONS>\n{dump(deps.observations)}\n</OBSERVATIONS>\n"
            f"\n<ACTIONS>\n{dump(deps.actions)}\n</ACTIONS>\n"
            "对照各角色的观察与行动, 裁决行动是否成功及其后果."
        )
    )


# ======================================================================================
# Simulation
# ======================================================================================
class Simulation:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        path: Path,
    ) -> None:
        self.path = path

    @property
    def latest_path(  # noqa: D102
        self,
    ) -> Path:
        versions = [
            path
            for path in self.path.iterdir()
            if path.is_dir() and re.fullmatch(r"ver_\d{13}", path.name)
        ]
        if len(versions) == 0:
            msg = f"No version folders found in {self.path}"
            raise FileNotFoundError(msg)
        return max(versions, key=lambda path: path.name)

    @staticmethod
    def read_memories(  # noqa: D102
        world: Path,
        owner: str | None = None,
    ) -> list[WorldMemory | CharacterMemory]:
        path = world / "memory.json"
        if not path.exists():
            return []

        adapter = TypeAdapter[WorldMemory | CharacterMemory](
            WorldMemory | CharacterMemory
        )

        memories = [
            adapter.validate_python(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]
        if owner is not None:
            memories = [m for m in memories if m.owner == owner]
            kind_order = {"OBSERVE": 0, "ACTION": 1, "EVENT": 2}
            memories.sort(key=lambda m: (m.datetime, kind_order[m.kind]))

        return memories

    @staticmethod
    def read_states(  # noqa: D102
        world: Path,
        owner: str | None = None,
    ) -> list[WorldState | CharacterState]:
        adapter = TypeAdapter[WorldState | CharacterState](WorldState | CharacterState)

        if owner is None:
            return [
                adapter.validate_python(json.loads(path.read_text(encoding="utf-8")))
                for path in (world / "state").iterdir()
                if path.is_file() and path.suffix == ".json"
            ]

        target = world / "state" / f"{owner}.json"
        if not target.exists():
            return []

        return [adapter.validate_python(json.loads(target.read_text(encoding="utf-8")))]

    @staticmethod
    def update_world(  # noqa: D102
        world: Path,
        updates: Sequence[WorldMemory | CharacterMemory | WorldState | CharacterState],
    ) -> None:
        memories = [u for u in updates if isinstance(u, (WorldMemory, CharacterMemory))]
        if len(memories) > 0:
            memories = Simulation.read_memories(world) + memories
            kind_order = {"OBSERVE": 0, "ACTION": 1, "EVENT": 2}
            memories.sort(key=lambda m: (m.datetime, kind_order[m.kind]))
            target = world / "memory.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    [m.model_dump(mode="json") for m in memories],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        for u in updates:
            if isinstance(u, (WorldState, CharacterState)):
                target = world / "state" / f"{u.owner}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        u.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    def run(  # noqa: D102
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 创建临时世界目录并复制最新版本的世界数据到该目录
            world = Path(tmp)
            shutil.copytree(self.latest_path, world, dirs_exist_ok=True)

            # INIT 阶段, GM 推进时间线
            init_result = executor.run_sync(
                deps=InitDeps(world=world),
                output_type=list[WorldMemory | WorldState | CharacterState],
            ).output
            self.update_world(world, init_result)

            # OBSERVE 阶段, GM 选择行动角色并生成其观察信息
            observe_result = executor.run_sync(
                deps=ObserveDeps(world=world),
                output_type=list[Observation],
            ).output

            # ACT 阶段, CHARACTER 根据观察信息生成行动
            act_result: list[Action] = [
                Action(
                    character=observation.character,
                    action=executor.run_sync(
                        deps=ActDeps(
                            world=world,
                            character=observation.character,
                            observation=observation.observation,
                        ),
                        output_type=str,
                    ).output,
                )
                for observation in observe_result
            ]

            # JUDGE 阶段, GM 裁决角色的行动
            judge_result = executor.run_sync(
                deps=JudgeDeps(
                    world=world,
                    actions=act_result,
                    observations=observe_result,
                ),
                output_type=list[
                    WorldMemory | CharacterMemory | WorldState | CharacterState
                ],
            ).output
            self.update_world(world, judge_result)

            # 将裁决后的世界状态保存为新版本
            shutil.copytree(world, self.path / _version())


# ======================================================================================
# Fields
# ======================================================================================
PathField = Annotated[
    Path,
    typer.Option(
        "--path",
        help="数据目录.",
    ),
]
RoundsField = Annotated[
    int,
    typer.Option(
        "--rounds",
        help="模拟轮数.",
    ),
]

# ======================================================================================
# Commands
# ======================================================================================
app = typer.Typer(
    name="simulation",
    help="虚拟世界模拟",
    no_args_is_help=True,
)


@app.command()
def demo(
    *,
    path: PathField,
    rounds: RoundsField,
) -> None:
    """启动并运行 demo."""
    if path.exists() and not path.is_dir():
        errmsg = f"Path is not a directory: {path!r}"
        raise typer.BadParameter(errmsg)

    if path.is_dir() and any(path.iterdir()):
        errmsg = f"Directory is not empty: {path!r}"
        raise typer.BadParameter(errmsg)

    Simulation.update_world(
        world=path / _version(),
        updates=[
            WorldState(
                owner="world",
                datetime=(now := datetime.now(tz=UTC)),
                description=(
                    "清晨, 迷雾森林边缘的小村庄刚刚苏醒, 村民们开始了一天的劳作."
                ),
            ),
            CharacterState(
                owner="alice",
                datetime=now,
                description=(
                    "Alice 是村庄的年轻草药师, 住在村口的小木屋里, 正准备外出采集草药."
                ),
                inventory="草药篮 x1, 银币 x10, 小刀 x1",
            ),
        ],
    )

    for _ in range(rounds):
        Simulation(path).run()


@app.command()
def run(
    *,
    path: PathField,
    rounds: RoundsField,
) -> None:
    """启动并运行虚拟世界."""
    if path.exists() and not path.is_dir():
        errmsg = f"Path is not a directory: {path!r}"
        raise typer.BadParameter(errmsg)

    for _ in range(rounds):
        Simulation(path).run()
