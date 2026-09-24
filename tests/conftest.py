import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_preprocessed_data


@pytest.fixture(scope="session")
def catalog():
    return get_preprocessed_data()
