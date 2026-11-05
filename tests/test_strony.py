from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

KATALOG = Path(__file__).resolve().parents[1]
STRONY = sorted(p.relative_to(KATALOG).as_posix() for p in (KATALOG / "strony").glob("*.py"))


@pytest.mark.parametrize("strona", STRONY)
def test_strona_bez_wyjatkow(strona):
    at = AppTest.from_file(str(KATALOG / "app.py"), default_timeout=300)
    at.run()
    at.switch_page(strona).run()
    assert not at.exception, [e.value for e in at.exception]
