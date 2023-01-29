import pytest
from drtsans.settings import namedtuplefy, amend_config
from mantid.kernel import ConfigService


@pytest.mark.offline
def test_namedtuplefy():
    @namedtuplefy
    def foo(x):
        return dict(foo=x)

    @namedtuplefy
    def goo(x):
        return dict(goo=x)

    y1 = foo(42)
    z1 = goo(24)
    y2 = foo(41)
    z2 = goo(21)

    assert type(y1) == type(y2)
    assert type(z1) == type(z2)
    assert type(y1) != type(z1)
    assert type(y2) != type(z2)


@pytest.mark.offline
def test_amend_config():
    config = ConfigService.Instance()
    old_instrument = config["instrumentName"]
    with amend_config({"instrumentName": "42"}):
        assert config["instrumentName"] == "42"
    assert config["instrumentName"] == old_instrument


@pytest.mark.offline
def test_offline():
    print("this tests runs when offline")


if __name__ == "__main__":
    pytest.main([__file__])
