import pytest
from os.path import join as pj
from drtsans.tof.eqsans import cfg


def test_setitem():
    c = cfg.Cfg()
    value = cfg.CfgItemValue(data=42, note="meaning of universe")
    c["k"] = value  # set value
    assert c["k"] == value  # get value


def test_rec_mask():
    # second rectangle is included in the first
    c = cfg.CfgItemRectangularMask(data=["0, 0; 1, 255", "1 ,250; 1, 255"])
    assert len(c.pixels) == 512
    assert (1, 255) in c.pixels
    # adding a mask already in mask `c` doesn't add any pixel
    d = cfg.CfgItemRectangularMask(data=["1 ,252; 1, 254"])
    c += d
    assert len(c.pixels) == 512
    # adding another tube to the mask
    d = cfg.CfgItemRectangularMask(data=["2 ,0; 2, 255"])
    c += d
    assert len(c.pixels) == 768


def test_ell_mask():
    c = cfg.CfgItemEllipticalMask(data=["1 ,250; 1, 255"])
    px = c.pixels
    assert (1, 255) in px


def test_mask_mixin():
    c = cfg.CfgItemRectangularMask(data=["1, 0; 2, 255", "1 250 1 255"])
    dets = c.detectors
    assert len(dets) == 2 * 256
    assert (
        dets[-1] == 5 * 256 - 1
    )  # x=1 corresponds to tube_index=4, thus five detectors.
    assert c.value == dets
    c = cfg.CfgItemEllipticalMask(data=["1, 250, 1, 255"])
    assert c.detectors == [1274, 1275, 1276, 1277, 1278, 1279]
    assert c.value == c.detectors


def test_tofedgediscard():
    c = cfg.CfgTofEdgeDiscard(data="500 2000")
    assert c.value == (500.0, 2000.0)


def test_closest_config(reference_dir):
    config_dir = pj(reference_dir.new.eqsans, "instrument_configuration")
    name = pj(config_dir, "eqsans_configuration.92474")
    assert cfg.closest_config(97711, config_dir=config_dir) == name


def test_open_source(reference_dir):
    config_dir = pj(reference_dir.new.eqsans, "instrument_configuration")
    name = "eqsans_configuration.92474"
    full_name = pj(config_dir, name)
    with cfg.open_source(full_name) as f:
        assert f.name == full_name
    with cfg.open_source(name, config_dir=config_dir) as f:
        assert f.name == full_name
    with cfg.open_source(97711, config_dir=config_dir) as f:
        assert f.name == full_name


def test_load(reference_dir):
    config_dir = pj(reference_dir.new.eqsans, "instrument_configuration")
    c = cfg.Cfg(source=97711, config_dir=config_dir)
    value = cfg.CfgItemValue(data="500 2000")
    assert c["tof edge discard"] == value
    assert isinstance(c["rectangular mask"], cfg.CfgItemRectangularMask)
    d = c.as_dict()
    assert d["rectangular mask"] == c["rectangular mask"].detectors


def test_load_config(reference_dir):
    config_dir = pj(reference_dir.new.eqsans, "instrument_configuration")
    d = cfg.load_config(source=97711, config_dir=config_dir)
    assert len(d["rectangular mask"]) == 7203
    assert "elliptical mask" not in d
    assert d["tof edge discard"] == (500.0, 2000.0)

    d = cfg.load_config(source=7554, config_dir=config_dir)
    assert len(d["rectangular mask"]) == 7088
    assert len(d["elliptical mask"]) == 270
    assert len(d["combined mask"]) == 7088 + 270


if __name__ == "__main__":
    pytest.main([__file__])
