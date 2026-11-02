from __future__ import annotations  # noqa: D100

from pathlib import Path
from typing import Annotated

import typer

from .model import CharacterState, WorldState
from .simulation import Simulation

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
        world=path / Simulation.create_version(),
        updates=[
            WorldState(
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
