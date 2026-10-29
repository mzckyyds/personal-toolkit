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
from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter, field_validator
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


def _sort_memories(
    memories: list[WorldMemory | CharacterMemory],
) -> list[WorldMemory | CharacterMemory]:
    kind_order = {"OBSERVATION": 0, "INTENTION": 1, "ACTION": 2, "EVENT": 3}
    return sorted(memories, key=lambda m: (m.ts, kind_order.get(m.kind, 99)))


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
# Data Model
# ======================================================================================
TimestampField = Field(
    description="时间戳, 带时区信息(默认东八区)的日期时间, ISO 8601 格式.",
)
WorldOwnerField = Field(
    description="归属者, 固定为 'world', 表示属于世界.",
)
CharacterOwnerField = Field(
    description="归属者, 不能为 'world', 表示属于指定角色.",
)
CharacterField = Field(
    description="角色名.",
)
ObservationField = Field(
    description="角色观察, 以角色第一人称视角记录, 如: '我看见了一个奇怪的物体'.",
)
IntentionField = Field(
    description="角色意图, 以角色第一人称视角记录, 如: '我决定离开这个地方'.",
)
ActionField = Field(
    description="角色行动, 以角色第一人称视角记录, 如: '我飞奔出去, 离开了这里'.",
)


class WorldMemory(BaseModel):
    """世界记忆, 记录世界发生的事件."""

    ts: AwareDatetime = TimestampField
    owner: Literal["world"] = WorldOwnerField
    kind: Literal["EVENT"] = Field(
        default="EVENT",
        description="类型, 固定为 'EVENT', 表示世界发生的事件.",
    )
    memory: str = Field(
        description="具体内容, 涉及角色时用第三人称角色名来记录, 否则直接客观记录.",
    )


class CharacterMemory(BaseModel):
    """角色记忆, 记录角色的观察/意图/行动."""

    ts: AwareDatetime = TimestampField
    owner: str = CharacterOwnerField
    kind: Literal["OBSERVATION", "INTENTION", "ACTION"] = Field(
        description="类型, 'OBSERVATION'/'INTENTION'/'ACTION' 分别表示观察/意图/行动.",
    )
    memory: str = Field(
        description="具体内容, 以角色第一人称视角记录, 如: "
        "'我看见了一个奇怪的物体', "
        "'我决定离开这个地方', "
        "'我飞奔出去, 离开了这里'."
    )

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


class WorldState(BaseModel):
    """世界状态."""

    ts: AwareDatetime = TimestampField
    owner: Literal["world"] = WorldOwnerField
    state: str = Field(
        description="世界状态的具体内容, 如世界的地理布局、天气状况、物资分布等.",
    )


class CharacterState(BaseModel):
    """角色状态."""

    ts: AwareDatetime = TimestampField
    owner: str = CharacterOwnerField
    state: str = Field(
        description="角色状态的具体内容, 如角色的属性、能力、道具、装备等.",
    )

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


class Observation(BaseModel):
    """角色观察."""

    character: str = CharacterField
    observation: str = ObservationField


class Intention(BaseModel):
    """角色意图."""

    character: str = CharacterField
    intention: str = IntentionField


class Action(BaseModel):
    """角色行动."""

    character: str = CharacterField
    action: str = ActionField


# ======================================================================================
# Executor
# ======================================================================================
@dataclass
class IncomeDeps:  # noqa: D101
    ts: AwareDatetime
    world: Path


@dataclass
class ObservationDeps:  # noqa: D101
    ts: AwareDatetime
    world: Path


@dataclass
class IntentionDeps:  # noqa: D101
    ts: AwareDatetime
    world: Path
    character: str
    observation: str


@dataclass
class ActionDeps:  # noqa: D101
    ts: AwareDatetime
    world: Path
    observations: list[Observation]
    intentions: list[Intention]


@dataclass
class OutcomeDeps:  # noqa: D101
    ts: AwareDatetime
    world: Path
    observations: list[Observation]
    intentions: list[Intention]
    actions: list[Action]


Deps = IncomeDeps | ObservationDeps | IntentionDeps | ActionDeps | OutcomeDeps


executor = Agent(
    model=OpenAIChatModel(
        os.environ.get("OEPNAI_LLM_NAME", ""),
        provider=OpenAIProvider(
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    ),
    deps_type=Deps,
    model_settings={"extra_body": {"thinking": {"type": "disabled"}}},
)


@executor.instructions
def instructions(  # noqa: D103
    ctx: RunContext[Deps],
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

    if isinstance(deps, IncomeDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world, "world")))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责推进世界时间线.")
            + task_segment.format(
                "阅读记忆与状态, 将时间线向前按需推进:\n"
                f"\n<DATETIME>\n{deps.ts}\n</DATETIME>"
            )
        )

    if isinstance(deps, ObservationDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world, "world")))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责挑选行动角色并生成观察.")
            + task_segment.format(
                "基于当前记忆与状态, 选出适合推进剧情的角色, "
                "并以角色第一人称视角生成观察内容."
            )
        )

    if isinstance(deps, IntentionDeps):
        character = deps.character
        return (
            memory_segment.format(dump(Simulation.read_memories(world, character)))
            + state_segment.format(dump(Simulation.read_states(world, character)))
            + role_segment.format(f"我是 {character}, 以第一人称思考与行动.")
            + task_segment.format(
                "我将基于当前观察给出我的意图:\n"
                f"\n<DATETIME>\n{deps.ts}\n</DATETIME>\n"
                f"\n<OBSERVATION>\n{dump(deps.observation)}\n</OBSERVATION>\n"
                "\n<INTENTION>\n{intention}\n</INTENTION>\n"
                "\nintention:\n"
            )
        )

    if isinstance(deps, ActionDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world, "world")))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责裁决各角色的行动结果.")
            + task_segment.format(
                "基于各角色的观察/意图, 裁决各角色的行动结果:\n"
                f"\n<DATETIME>\n{deps.ts}\n</DATETIME>\n"
                f"\n<OBSERVATIONS>\n{dump(deps.observations)}\n</OBSERVATIONS>\n"
                f"\n<INTENTIONS>\n{dump(deps.intentions)}\n</INTENTIONS>\n"
                "\n<ACTIONS>\n{actions}\n</ACTIONS>\n"
                "\nactions:\n"
            )
        )

    return (
        memory_segment.format(dump(Simulation.read_memories(world, "world")))
        + state_segment.format(dump(Simulation.read_states(world)))
        + role_segment.format("你是游戏主持人(GM), 负责推进游戏.")
        + task_segment.format(
            "基于各角色的观察/意图/行动, 推进游戏:\n"
            f"\n<DATETIME>\n{deps.ts}\n</DATETIME>\n"
            f"\n<OBSERVATIONS>\n{dump(deps.observations)}\n</OBSERVATIONS>\n"
            f"\n<INTENTIONS>\n{dump(deps.intentions)}\n</INTENTIONS>\n"
            f"\n<ACTIONS>\n{dump(deps.actions)}\n</ACTIONS>\n"
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

        return _sort_memories(memories)

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
            memories = _sort_memories(Simulation.read_memories(world) + memories)
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

            # 获取当前世界状态的时间戳
            ts = self.read_states(world, "world")[0].ts

            # INCOME 阶段, GM 推进时间线
            incomes = executor.run_sync(
                deps=IncomeDeps(
                    ts=ts,
                    world=world,
                ),
                output_type=list[WorldMemory | WorldState | CharacterState],
            ).output
            self.update_world(
                world=world,
                updates=incomes,
            )

            # 获取推进后的时间戳
            ts = self.read_states(world, "world")[0].ts

            # OBSERVATION 阶段, GM 选择行动角色并为其生成观察信息
            observations = executor.run_sync(
                deps=ObservationDeps(
                    ts=ts,
                    world=world,
                ),
                output_type=list[Observation],
            ).output

            # INTENTION 阶段, CHARACTER 根据观察信息产生行动意图
            intentions: list[Intention] = [
                Intention(
                    character=observation.character,
                    intention=executor.run_sync(
                        deps=IntentionDeps(
                            ts=ts,
                            world=world,
                            character=observation.character,
                            observation=observation.observation,
                        ),
                        output_type=str,
                    ).output,
                )
                for observation in observations
            ]

            # ACTION 阶段, GM 根据角色的行动意图生成实际行动
            actions: list[Action] = executor.run_sync(
                deps=ActionDeps(
                    ts=ts,
                    world=world,
                    observations=observations,
                    intentions=intentions,
                ),
                output_type=list[Action],
            ).output

            # OUTCOME 阶段, GM 根据角色的观察/意图/实际行动生成裁决结果
            outcomes = executor.run_sync(
                deps=OutcomeDeps(
                    ts=ts,
                    world=world,
                    observations=observations,
                    intentions=intentions,
                    actions=actions,
                ),
                output_type=list[WorldMemory | WorldState | CharacterState],
            ).output
            self.update_world(
                world,
                [
                    CharacterMemory(
                        ts=ts,
                        kind="OBSERVATION",
                        owner=o.character,
                        memory=o.observation,
                    )
                    for o in observations
                ]
                + [
                    CharacterMemory(
                        ts=ts,
                        kind="INTENTION",
                        owner=i.character,
                        memory=i.intention,
                    )
                    for i in intentions
                ]
                + [
                    CharacterMemory(
                        ts=ts,
                        kind="ACTION",
                        owner=a.character,
                        memory=a.action,
                    )
                    for a in actions
                ]
                + outcomes,
            )

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
                ts=(now := datetime.now(tz=UTC)),
                owner="world",
                state="清晨, 迷雾森林边缘的小村庄刚刚苏醒, 村民们开始了一天的劳作.",
            ),
            CharacterState(
                ts=now,
                owner="alice",
                state="Alice 是村庄的年轻草药师, 住在村口的小木屋里, 正准备外出采药.",
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
