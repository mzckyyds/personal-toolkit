from __future__ import annotations  # noqa: D100

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .model import (
    Action,
    CharacterMemory,
    CharacterState,
    Intention,
    Observation,
    WorldMemory,
    WorldState,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Simulation",
]


def _dump(obj: object) -> str:
    return json.dumps(
        obj, ensure_ascii=False, default=lambda o: o.model_dump(mode="json")
    )


def _sort_memories(
    memories: list[WorldMemory | CharacterMemory],
) -> list[WorldMemory | CharacterMemory]:
    kind_order = {"OBSERVATION": 0, "ACTION": 1, "EVENT": 2}
    return sorted(memories, key=lambda m: (m.ts, kind_order.get(m.kind, 99)))


def _load_prompt(
    tag: Literal["observation", "intention", "action", "outcome"],
) -> str:
    return (Path(__file__).parent / "prompt" / f"{tag}.prompt").read_text(
        encoding="utf-8"
    )


class Simulation:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        path: Path,
    ) -> None:
        self.path = path
        self.agent = Agent(
            model=OpenAIChatModel(
                os.environ.get("OPENAI_LLM_NAME", ""),
                provider=OpenAIProvider(
                    base_url=os.environ.get("OPENAI_BASE_URL", ""),
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                ),
            ),
            model_settings={"extra_body": {"thinking": {"type": "disabled"}}},
        )

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
    def create_version(  # noqa: D102
    ) -> str:
        return f"ver_{time.time_ns() // 1_000_000}"

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

            world_memory = _dump(self.read_memories(world, "world"))
            world_state = _dump(self.read_states(world))

            # OBSERVATION 阶段, GM 选择角色并为其生成`观察信息`
            observations = self.agent.run_sync(
                instructions=_load_prompt("observation").format(
                    memory=world_memory,
                    state=world_state,
                ),
                output_type=list[Observation],
            ).output

            # INTENTION 阶段, CHARACTER 根据`观察信息`产生`行动意图`
            intentions: list[Intention] = [
                Intention(
                    ts=observation.ts,
                    character=observation.character,
                    intention=self.agent.run_sync(
                        instructions=_load_prompt("intention").format(
                            memory=_dump(
                                self.read_memories(world, observation.character)
                            ),
                            state=_dump(self.read_states(world, observation.character)),
                            observation=_dump(observation.observation),
                            character=observation.character,
                        ),
                        output_type=str,
                    ).output,
                )
                for observation in observations
            ]

            # ACTION 阶段, GM 根据角色的`观察信息`/`行动意图`, 裁决其`实际行动`
            actions: list[Action] = self.agent.run_sync(
                instructions=_load_prompt("action").format(
                    memory=world_memory,
                    state=world_state,
                    observations=_dump(observations),
                    intentions=_dump(intentions),
                ),
                output_type=list[Action],
            ).output

            # OUTCOME 阶段, GM 根据角色的`观察信息`/`行动意图`/`实际行动`, 推进游戏
            outcomes = self.agent.run_sync(
                instructions=_load_prompt("outcome").format(
                    memory=world_memory,
                    state=world_state,
                    observations=_dump(observations),
                    intentions=_dump(intentions),
                    actions=_dump(actions),
                ),
                output_type=list[WorldMemory | WorldState | CharacterState],
            ).output
            self.update_world(
                world,
                [
                    CharacterMemory(
                        ts=o.ts,
                        kind="OBSERVATION",
                        owner=o.character,
                        memory=o.observation,
                    )
                    for o in observations
                ]
                + [
                    CharacterMemory(
                        ts=a.ts,
                        kind="ACTION",
                        owner=a.character,
                        memory=a.action,
                    )
                    for a in actions
                ]
                + outcomes,
            )

            # 将裁决后的世界状态保存为新版本
            shutil.copytree(world, self.path / self.create_version())
