from __future__ import annotations  # noqa: D100

from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

__all__ = [
    "Action",
    "CharacterMemory",
    "CharacterState",
    "Intention",
    "Observation",
    "WorldMemory",
    "WorldState",
]

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
    description="角色观察信息, 以角色第一人称视角记录, 仅包含角色通过感官"
    "(视觉/听觉/嗅觉/触觉等)感知到的外部客观信息, 如: '我看见了一个奇怪的物体'. "
    "不得包含角色的行动/心理活动/决定/对自己的评述, 也不得包含角色感知不到的信息.",
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

    owner: Literal["world"] = WorldOwnerField
    state: str = Field(
        description="世界状态的具体内容, 如世界的地理布局、天气状况、物资分布等.",
    )


class CharacterState(BaseModel):
    """角色状态."""

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

    ts: AwareDatetime = TimestampField
    character: str = CharacterField
    observation: str = ObservationField


class Intention(BaseModel):
    """角色行动意图."""

    ts: AwareDatetime = TimestampField
    character: str = CharacterField
    intention: str = IntentionField


class Action(BaseModel):
    """角色实际行动."""

    ts: AwareDatetime = TimestampField
    character: str = CharacterField
    action: str = ActionField
