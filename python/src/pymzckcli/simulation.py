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
    kind_order = {"OBSERVATION": 0, "ACTION": 1, "EVENT": 2}
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
    description="角色观察信息, 以角色第一人称视角记录, 如: '我看见了一个奇怪的物体'.",
)
IntentionField = Field(
    description="角色行动意图, 以角色第一人称视角记录, 如: '我决定离开这个地方'.",
)
ActionField = Field(
    description="角色实际行动, 以角色第一人称视角记录, 如: '我飞奔出去, 离开了这里'.",
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
    """角色记忆, 记录角色的观察信息/实际行动."""

    ts: AwareDatetime = TimestampField
    owner: str = CharacterOwnerField
    kind: Literal["OBSERVATION", "ACTION"] = Field(
        description="类型, 'OBSERVATION'/'ACTION' 分别表示观察信息/实际行动.",
    )
    memory: str = Field(
        description="具体内容, 以角色第一人称视角记录, 如: "
        "'我看见了一个奇怪的物体', "
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
    """角色观察信息."""

    character: str = CharacterField
    observation: str = ObservationField


class Intention(BaseModel):
    """角色行动意图."""

    character: str = CharacterField
    intention: str = IntentionField


class Action(BaseModel):
    """角色实际行动."""

    character: str = CharacterField
    action: str = ActionField


# ======================================================================================
# Executor
# ======================================================================================
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


Deps = ObservationDeps | IntentionDeps | ActionDeps | OutcomeDeps


executor = Agent(
    model=OpenAIChatModel(
        os.environ.get("OPENAI_LLM_NAME", ""),
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

    if isinstance(deps, ObservationDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world, "world")))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责选择角色并为其生成观察信息.")
            + task_segment.format(
                "基于当前记忆与状态, 选出适合推进剧情的角色, "
                "并以角色第一人称视角生成观察信息."
            )
        )

    if isinstance(deps, IntentionDeps):
        character = deps.character
        return (
            memory_segment.format(dump(Simulation.read_memories(world, character)))
            + state_segment.format(dump(Simulation.read_states(world, character)))
            + role_segment.format(f"我是 {character}, 以第一人称思考与行动.")
            + task_segment.format(
                "我将基于当前观察信息给出我的行动意图:\n"
                f"\n<DATETIME>\n{deps.ts}\n</DATETIME>\n"
                f"\n<OBSERVATION>\n{dump(deps.observation)}\n</OBSERVATION>\n"
                "\n<INTENTION>\n{行动意图}\n</INTENTION>\n"
                "\n行动意图:\n"
            )
        )

    if isinstance(deps, ActionDeps):
        return (
            memory_segment.format(dump(Simulation.read_memories(world, "world")))
            + state_segment.format(dump(Simulation.read_states(world)))
            + role_segment.format("你是游戏主持人(GM), 负责裁决各角色的实际行动.")
            + task_segment.format(
                "基于各角色的观察信息/行动意图, 裁决各角色的实际行动:\n"
                f"\n<DATETIME>\n{deps.ts}\n</DATETIME>\n"
                f"\n<OBSERVATIONS>\n{dump(deps.observations)}\n</OBSERVATIONS>\n"
                f"\n<INTENTIONS>\n{dump(deps.intentions)}\n</INTENTIONS>\n"
                "\n<ACTIONS>\n{实际行动}\n</ACTIONS>\n"
                "\n实际行动:\n"
            )
        )

    return (
        memory_segment.format(dump(Simulation.read_memories(world, "world")))
        + state_segment.format(dump(Simulation.read_states(world)))
        + role_segment.format("你是游戏主持人(GM), 负责推进游戏.")
        + task_segment.format(
            "基于各角色的观察信息/行动意图/实际行动, 推进游戏:\n"
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

            # OBSERVATION 阶段, GM 选择角色并为其生成`观察信息`
            observations = executor.run_sync(
                deps=ObservationDeps(
                    ts=ts,
                    world=world,
                ),
                output_type=list[Observation],
            ).output

            # INTENTION 阶段, CHARACTER 根据`观察信息`产生`行动意图`
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

            # ACTION 阶段, GM 根据角色的`观察信息`/`行动意图`, 裁决其`实际行动`
            actions: list[Action] = executor.run_sync(
                deps=ActionDeps(
                    ts=ts,
                    world=world,
                    observations=observations,
                    intentions=intentions,
                ),
                output_type=list[Action],
            ).output

            # OUTCOME 阶段, GM 根据角色的`观察信息`/`行动意图`/`实际行动`, 推进游戏
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
                state="""实验性 X 药剂泄漏并扩散全球, 大量人类感染为丧尸.
丧尸逐渐聚集为尸潮.
社会秩序已经崩溃, 幸存者在各个集结点依靠防御设施求生, 通过无线电维持有限联系.
当前为生存挑战第 1 天. 0 号集结点防线刚刚失守, 撤离警报响起.
广播反复通知: '所有幸存者立即前往仍在运作的 111 号集结点!'
阿帧正在 0 号集结点的临时住处, 刚从睡梦中被警报惊醒, 尚未采取行动.
前往 111 号集结点的沿途有零散丧尸, 局部尸群, 道路障碍和其他幸存者.
身后的尸潮不断向前推进. 绕路, 搜寻, 休息和帮助他人都需要耗时, 会影响撤离机会.
111 号集结点仍在运作, 幸存者到达后还需排队, 登记, 完成感染检查与接收.
当前阶段目标是在尸潮抵达之前完成接收, 但抵达集结点不代表永久安全.
这是 AI 漫剧主播阿帧搭建的自动演化虚拟世界, 也是一场存活满 30 天的直播挑战.
直播不依赖角色携带拍摄设备. 观众评论可作为游戏系统的干预输入, 带来帮助或整活事件.
当前尚未收到观众评论. 干预应符合现有世界条件, 不直接替角色决定思想和行动,
不无故改写已发生的事实, 也不直接宣判角色死亡或通关.
角色没有管理员权限, 隐藏视野或死亡读档能力. 世界事实不自动成为角色已知信息.
所有行动都有合理耗时, 资源, 伤势, 位置和人物关系持续演化, 不提供主角保护.
最终目标是在世界中存活满 30 天, 故事由角色选择, 客观条件和观众干预自然产生.""",
            ),
            CharacterState(
                ts=now,
                owner="azhen",
                state="""我叫阿帧, 是个做 AI 漫剧的主播. 出图, 配音, 剪辑我都会折腾,
唯独写不好剧本. 后来我想通了: 既然不会编故事, 那就造一个能自己发生故事的世界.
于是我搭建了一个自动演化的虚拟世界, 把自己的数字化身放进去, 发起了一场直播挑战:
'你们发评论, 系统负责整活, 我负责活满三十天.'
现在我就是身处这个世界里的阿帧. 我知道自己正在参加挑战, 也知道观众能通过系统影响世界,
但我没有后台权限, 不能暂停, 读档或凭空生成物资, 也看不到隐藏地图和他人的内心.
我是普通成年人, 没受过专业的战斗, 医疗或野外生存训练. 我习惯观察细节, 联想和临场表达,
但做视频的经验不等于求生本领. 我喜欢吐槽, 有点嘴硬, 紧张时话可能更多.
真遇到危险, 我也会害怕, 犹豫, 犯错, 并根据实际反馈调整判断.
我想做出节目效果, 更想活下来.
我不一定知道哪些遭遇是观众搞出来的. 即使收到系统通知, 我也得自己判断要不要相信和照做.
我刚在临时住处被撤离警报惊醒, 听见广播说 0 号集结点防线失守, 要立即前往 111 号集结点.
我还不知道外面具体有多危险, 哪条路能走, 尸潮离我多远, 也尚未决定下一步行动.
我身体健康, 没有感染或伤病, 体力充足, 没有明显饥饿, 口渴或疲劳.
除身上的普通衣物外, 我只有一瓶 500 毫升矿泉水、一个 100 克面包和一辆能正常骑行的自行车,
没有手机, 背包, 武器, 药品或其他装备, 也没有特殊能力.
我的长期目标是在这里活满 30 天. 眼下的目标是在尸潮抵达前赶到 111 号集结点,
并花时间完成检查与接收. 到了那里以后怎么办, 我现在还没有答案.""",
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
