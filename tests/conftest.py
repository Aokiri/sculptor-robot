import matplotlib
import pytest

matplotlib.use("Agg")

from experiments.fake_log import make_fake_log


@pytest.fixture
def sample_log():
    return make_fake_log()
