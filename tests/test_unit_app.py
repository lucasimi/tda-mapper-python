import asyncio
from dataclasses import asdict

from nicegui.testing import User
from sklearn.datasets import load_digits

from tdamapper import app

RETRIES = 40

pytest_plugins = ["nicegui.testing.user_plugin"]


async def test_run_app_fail(user: User) -> None:
    app.startup()
    await user.open("/")
    await user.should_see("Load Data")
    await user.should_see("Lens")
    await user.should_see("Cover")
    await user.should_see("Clustering")
    await user.should_see("Run Mapper")
    await user.should_see("Redraw")
    user.find("Run Mapper").click()
    await user.should_see("Run Mapper failed")
    await user.should_not_see("Load data completed")


async def test_run_app_success(user: User) -> None:
    app.startup()
    await user.open("/")
    await user.should_see("Load Data")
    await user.should_see("Lens")
    await user.should_see("Cover")
    await user.should_see("Clustering")
    await user.should_see("Run Mapper")
    await user.should_see("Redraw")

    # Click on the toggle menu button to open the menu
    await user.should_see("menu")
    user.find("menu").click()
    await user.should_see("menu")
    user.find("menu").click()

    user.find("Load Data").click()
    await user.should_see("Load data completed")
    await user.should_not_see("Load data failed")
    user.find("Run Mapper").click()
    await user.should_see("Running Mapper...")
    await user.should_not_see("Run Mapper failed")
    await user.should_see("Run Mapper completed", retries=RETRIES)
    await user.should_see("Drawing Mapper...")
    await user.should_not_see("Draw Mapper failed")
    await user.should_see("Draw Mapper completed", retries=RETRIES)
    user.find("Redraw").click()
    await user.should_see("Drawing Mapper...")
    await user.should_not_see("Draw Mapper failed")
    await user.should_see("Draw Mapper completed", retries=RETRIES)


def test_run_mapper() -> None:
    config = app.MapperConfig()
    df_X, df_target = load_digits(return_X_y=True, as_frame=True)
    result = app.run_mapper(df_X, **asdict(config))
    assert result is not None
    mapper_graph, df_y = result
    mapper_fig = app.create_mapper_figure(
        df_X, df_y, df_target, mapper_graph, **asdict(config)
    )
    assert mapper_fig is not None
