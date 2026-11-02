from __future__ import annotations  # noqa: D100

import json
import os
import re
import shutil
import tempfile
import time
from html import escape
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


def _sort_memories(
    memories: list[WorldMemory | CharacterMemory],
) -> list[WorldMemory | CharacterMemory]:
    kind_order = {"OBSERVATION": 0, "ACTION": 1, "EVENT": 2}
    return sorted(memories, key=lambda m: (m.ts, kind_order.get(m.kind, 99)))


def _load_prompt(
    tag: Literal["observation", "intention", "action", "outcome"],
    **kwargs: str,
) -> str:
    template = (Path(__file__).parent / "prompt" / f"{tag}.prompt").read_text(
        encoding="utf-8"
    )
    prompt = template.format(**kwargs)

    print("=" * 88)
    print(prompt)

    return prompt


class Simulation:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        *,
        path: Path,
        backstory: dict[str, str],
    ) -> None:
        self.path = path
        self.backstory = backstory.copy()
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

        self.path.mkdir(parents=True, exist_ok=True)
        if not any(
            child.is_dir() and re.fullmatch(r"ver_\d{13}", child.name)
            for child in self.path.iterdir()
        ):
            (self.path / Simulation.create_version() / "state").mkdir(parents=True)

    def _format_world(
        self,
        *,
        world: Path,
    ) -> str:
        parts = ["<WORLD>"]

        # Backstory for the world
        if (backstory := self.backstory.get("world", "")) != "":
            parts.append(f"<BACKSTORY>\n{escape(backstory, quote=False)}\n</BACKSTORY>")
        else:
            errmsg = "Missing backstory for 'world'"
            raise ValueError(errmsg)
        # Memories for the world
        if len(memories := Simulation.read_memories(world, "world")) > 0:
            parts.append(
                "\n".join(
                    f'<MEMORY timestamp="{escape(memory.ts.isoformat(), quote=True)}" '
                    f'kind="{escape(memory.kind, quote=True)}">'
                    f"\n{escape(memory.memory, quote=False)}\n</MEMORY>"
                    for memory in memories
                )
            )
        # States for the world
        if len(states := Simulation.read_states(world, "world")) == 1:
            parts.append(f"<STATE>\n{escape(states[0].state, quote=False)}\n</STATE>")
        elif len(states) > 1:
            errmsg = f"Expected one or zero world states in {world}, got {len(states)}"
            raise ValueError(errmsg)

        parts.append("</WORLD>")

        return "\n".join(parts)

    def _format_character(  # noqa: PLR0913
        self,
        *,
        world: Path,
        name: str,
        observation: Observation | None = None,
        intention: Intention | None = None,
        action: Action | None = None,
        with_memories: bool = False,
    ) -> str:
        parts = [f'<CHARACTER name="{escape(name, quote=True)}">']

        # Backstory for the character
        if (backstory := self.backstory.get(name, "")) != "":
            parts.append(f"<BACKSTORY>\n{escape(backstory, quote=False)}\n</BACKSTORY>")
        else:
            errmsg = f"Missing backstory for character {name!r}"
            raise ValueError(errmsg)
        # Memories for the character
        if with_memories:
            parts.extend(
                f'<MEMORY timestamp="{escape(memory.ts.isoformat(), quote=True)}" '
                f'kind="{escape(memory.kind, quote=True)}">'
                f"\n{escape(memory.memory, quote=False)}\n</MEMORY>"
                for memory in Simulation.read_memories(world, name)
            )
        # States for the character
        if len(states := Simulation.read_states(world, name)) == 1:
            parts.append(f"<STATE>\n{escape(states[0].state, quote=False)}\n</STATE>")
        elif len(states) > 1:
            errmsg = (
                f"Expected one or zero character states for {name} "
                f"in {world}, got {len(states)}"
            )
            raise ValueError(errmsg)
        # Observations, intentions, and actions for the character
        for tag, ts, text in (
            (
                "OBSERVATION",
                observation and observation.ts,
                observation and observation.observation,
            ),
            (
                "INTENTION",
                intention and intention.ts,
                intention and intention.intention,
            ),
            ("ACTION", action and action.ts, action and action.action),
        ):
            if ts is not None and text is not None:
                parts.append(
                    f'<{tag} timestamp="{escape(ts.isoformat(), quote=True)}">'
                    f"\n{escape(text, quote=False)}\n</{tag}>"
                )

        parts.append("</CHARACTER>")

        return "\n".join(parts)

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

            world_context = self._format_world(world=world)

            # OBSERVATION 阶段, GM 选择角色并为其生成`观察信息`
            observations = self.agent.run_sync(
                instructions=_load_prompt(
                    tag="observation",
                    world=world_context,
                    characters="\n".join(
                        [
                            self._format_character(
                                world=world,
                                name=name,
                            )
                            for name in self.backstory
                            if name != "world"
                        ]
                    ),
                ),
                output_type=list[Observation],
            ).output

            # INTENTION 阶段, CHARACTER 根据`观察信息`产生`行动意图`
            intentions: list[Intention] = [
                Intention(
                    ts=observation.ts,
                    character=observation.character,
                    intention=self.agent.run_sync(
                        instructions=_load_prompt(
                            "intention",
                            character=self._format_character(
                                world=world,
                                name=observation.character,
                                observation=observation,
                                with_memories=True,
                            ),
                        ),
                        output_type=str,
                    ).output,
                )
                for observation in observations
            ]

            # ACTION 阶段, GM 根据角色的`观察信息`/`行动意图`, 裁决其`实际行动`
            _intention_map = {
                intention.character: intention for intention in intentions
            }
            actions: list[Action] = self.agent.run_sync(
                instructions=_load_prompt(
                    "action",
                    world=world_context,
                    characters="\n".join(
                        self._format_character(
                            world=world,
                            name=observation.character,
                            observation=observation,
                            intention=_intention_map.get(observation.character),
                        )
                        for observation in observations
                    ),
                ),
                output_type=list[Action],
            ).output

            # OUTCOME 阶段, GM 根据角色的`观察信息`/`行动意图`/`实际行动`, 推进游戏
            _action_map = {action.character: action for action in actions}
            outcomes = self.agent.run_sync(
                instructions=_load_prompt(
                    "outcome",
                    world=world_context,
                    characters="\n".join(
                        self._format_character(
                            world=world,
                            name=observation.character,
                            observation=observation,
                            intention=_intention_map.get(observation.character),
                            action=_action_map.get(observation.character),
                        )
                        for observation in observations
                    ),
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
