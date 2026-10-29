"""Python CLI Toolkit."""

import typer

from .image import app as image_app
from .simulation import app as simulation_app
from .video import app as video_app
from .voice import app as voice_app

app = typer.Typer(
    help="Python CLI Toolkit",
    no_args_is_help=True,
    rich_markup_mode=None,
)

app.add_typer(voice_app)
app.add_typer(image_app)
app.add_typer(video_app)
app.add_typer(simulation_app)


def main() -> None:
    """入口函数."""
    app()


if __name__ == "__main__":
    main()
