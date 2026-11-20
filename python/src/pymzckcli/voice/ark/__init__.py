"""火山引擎语音识别/合成工具.

密钥使用环境变量 ARK_API_KEY.
"""

import typer

from .asr import recognize
from .tts import synthesize

__all__ = [
    "app",
]


app = typer.Typer(
    name="ark",
    help="火山引擎语音识别/合成工具",
    no_args_is_help=True,
)
app.command(name="asr")(recognize)
app.command(name="tts")(synthesize)
