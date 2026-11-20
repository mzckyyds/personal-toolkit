"""视频生成/超分工具."""

import typer

from .ucloud import app as ucloud_app

__all__ = [
    "app",
]


app = typer.Typer(
    name="video",
    help="视频生成/超分工具",
    no_args_is_help=True,
)

app.add_typer(ucloud_app)
