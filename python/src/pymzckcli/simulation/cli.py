from __future__ import annotations  # noqa: D100

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from .simulation import Simulation

__all__ = [
    "app",
]

_WORLD_BACKSTORY = """
某实验性“X药剂”泄漏并扩散至全球, 大量人类感染成丧尸, 并逐渐聚集形成尸潮. 随着
社会秩序崩溃, 幸存者分散聚居于各个集结点, 依靠防御设施抵御丧尸袭击, 并通过无线电
维持有限联系. 故事开始于0号集结点防线失守, 撤离警报响起, 广播要求幸存者立即前往
仍在运作的111号集结点. 所有角色开局相同: 每人带一瓶水、一块面包, 已经骑上各自的
普通自行车开始撤离, 不再争抢或挑选起始车辆.

撤离途中, 幸存者不仅要躲避身后持续推进的尸潮, 还要应对沿途的零散丧尸、局部尸群、
道路障碍及其他幸存者. 幸存者可以通过搜集、交易、合作等方式获取食物、饮水、燃油及
其他必需品, 同时需要维持自身健康与车辆状态.

世界时间持续推进, 移动、搜集、休息、修理等行动均会消耗时间, 停留过久可能被尸潮追
上, 但合理的侦察、清理和防御也能换来暂时的安全.

所有角色的共同目标是在尸潮抵达之前安全到达111号集结点并完成接收.
"""

_AZHEN_BACKSTORY = """
我叫阿帧, 是个AI漫剧创作者. 那天我正坐在电脑前制作漫剧, 一阵困意突然袭来,
我趴在桌上睡了过去. 再睁眼时, 熟悉的工作室消失了, 耳边全是警报与嘶吼.
我竟穿越到了丧尸横行的末日世界. 我不知道自己为什么来到这里, 也不知道怎么回去.
眼下只能先想办法活下来.
"""


# ======================================================================================
# Fields
# ======================================================================================
PathField = Annotated[
    Path,
    typer.Option(
        "--path",
        help="数据目录.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
]
BackstoryField = Annotated[
    Path | None,
    typer.Option(
        "--backstory",
        help="背景故事.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]
RoundsField = Annotated[
    int,
    typer.Option(
        "--rounds",
        help="模拟轮数.",
        min=0,
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
def run(
    *,
    path: PathField,
    rounds: RoundsField,
    backstory: BackstoryField = None,
) -> None:
    """启动或继续运行虚拟世界."""
    if rounds > 0:
        simulation = Simulation(
            path=path,
            backstory={
                "world": _WORLD_BACKSTORY,
                "阿帧": _AZHEN_BACKSTORY,
            }
            if backstory is None
            else TypeAdapter(dict[str, str]).validate_json(
                backstory.read_text(encoding="utf-8")
            ),
        )
        for _ in range(rounds):
            simulation.run()
