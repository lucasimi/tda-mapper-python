from dataclasses import asdict

from nicegui.testing import User
from sklearn.datasets import load_digits

from tdamapper import app

pytest_plugins = ["nicegui.testing.user_plugin"]


async def test_run_app(user: User) -> None:
    app.startup()
    await user.open("/")
    await user.should_see("Load Data")
    await user.should_see("Lens")
    await user.should_see("Cover")
    await user.should_see("Clustering")
    await user.should_see("Run Mapper")
    await user.should_see("Redraw")
    user.find("Load Data").click()
    await user.should_see("File loaded successfully")
    await user.should_not_see("No data found.")
    user.find("Run Mapper").click()
    await user.should_see("Running Mapper...")
    await user.should_not_see("No data found.")
    await user.should_see("Running Mapper Completed!", retries=20)
    await user.should_see("Drawing Mapper...")
    await user.should_not_see("No data")
    await user.should_see("Drawing Mapper Completed!", retries=20)
    user.find("Redraw").click()
    await user.should_see("Drawing Mapper...")
    await user.should_not_see("No data")
    await user.should_see("Drawing Mapper Completed!", retries=20)


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
