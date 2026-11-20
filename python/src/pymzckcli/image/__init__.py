"""图片生成/编辑工具."""

import typer

from .openai import app as openai_app

__all__ = [
    "app",
]


app = typer.Typer(
    name="image",
    help="图片生成/编辑工具",
    no_args_is_help=True,
)

app.add_typer(openai_app)
