import os

import pytest

CACHE = os.environ.get("XSUM_CACHE", os.path.join(os.path.dirname(__file__), "..", "data", "cache"))


@pytest.fixture(scope="session")
def cache():
    if not os.path.exists(os.path.join(CACHE, "mcnu.parquet")):
        pytest.skip(f"no slimmed cache at {CACHE}; run scripts/slim_cohpi_df.py")
    from xsum.io.cohpi_df import load_cache
    return load_cache(CACHE, ["mcnu", "evt"])
