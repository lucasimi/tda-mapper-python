import pytest
from nicegui.testing import User

from tdamapper import app

pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.mark.module_under_test(app)
async def test_run_mapper(user: User) -> None:
    await user.open("/")
    await user.should_see("Load Data")
    await user.should_see("Run Mapper")
    user.find("Load Data").click()
    await user.should_see("File loaded successfully")
    user.find("Run Mapper").click()
    await user.should_see("Running Mapper...")
