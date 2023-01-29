from collections import namedtuple
from os.path import join as pjoin
import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from mantid.simpleapi import Load, CreateWorkspace, CreateSampleWorkspace
from mantid.kernel import DateAndTime
from drtsans import wavelength as sans_wavelength
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans import correct_frame
from drtsans.settings import amend_config, unique_workspace_name
from drtsans.geometry import source_detector_distance

BandsTuple = namedtuple("BandsTuple", "lead skip")


def test_transmitted_bands(reference_dir):
    with amend_config(data_dir=reference_dir.new.eqsans):
        ws = Load(Filename="EQSANS_86217.nxs.h5")
        bands = correct_frame.transmitted_bands(ws)
        assert_almost_equal((bands.lead.min, bands.lead.max), (2.48, 6.78), decimal=2)
        assert_almost_equal((bands.skip.min, bands.skip.max), (10.90, 15.23), decimal=2)


def test_transmitted_bands_clipped(reference_dir):
    with amend_config(data_dir=reference_dir.new.eqsans):
        ws = Load(Filename="EQSANS_86217.nxs.h5")
        sdd = source_detector_distance(ws, unit="m")
        bands_0 = correct_frame.transmitted_bands_clipped(ws, sdd, 0.0, 0.0)
        lwc, hwc = (0.139, 0.560)  # expected clippings
        # With no interior clipping
        bands = correct_frame.transmitted_bands_clipped(
            ws, sdd, 500, 2000, interior_clip=False
        )
        # Check clippings for the lead pulse
        b1_0, b2_0 = bands_0.lead.min, bands_0.lead.max
        b1, b2 = bands.lead.min, bands.lead.max
        assert (b1, b2) == approx((b1_0 + lwc, b2_0), 0.01)
        # Check clippings for the skip pulse
        b1_0, b2_0 = bands_0.skip.min, bands_0.skip.max
        b1, b2 = bands.skip.min, bands.skip.max
        assert (b1, b2) == approx((b1_0, b2_0 - hwc), 0.01)
        # With interior clipping
        bands = correct_frame.transmitted_bands_clipped(
            ws, sdd, 500, 2000, interior_clip=True
        )
        b1_0, b2_0 = bands_0.lead.min, bands_0.lead.max
        b1, b2 = bands.lead.min, bands.lead.max
        assert (b1, b2) == approx((b1_0 + lwc, b2_0 - hwc), 0.01)
        b1_0, b2_0 = bands_0.skip.min, bands_0.skip.max
        b1, b2 = bands.skip.min, bands.skip.max
        assert (b1, b2) == approx((b1_0 + lwc, b2_0 - hwc), 0.01)


@pytest.mark.offline
def test_log_tof_structure(reference_dir):
    file_name = pjoin(
        reference_dir.new.eqsans, "test_correct_frame", "EQSANS_92353_no_events.nxs"
    )
    for ny, refv in ((False, 30833), (True, 28333)):
        ws = Load(file_name, OutputWorkspace=unique_workspace_name())
        correct_frame.log_tof_structure(ws, 500, 2000, interior_clip=ny)
        sl = SampleLogs(ws)
        assert sl.tof_frame_width.value == approx(33333, abs=1.0)
        assert sl.tof_frame_width_clipped.value == approx(refv, abs=1.0)
        ws.delete()


def test_band_structure_logs():
    w = CreateWorkspace([0], [0], OutputWorkspace=unique_workspace_name())
    with pytest.raises(RuntimeError, match="Band structure not found in the logs"):
        correct_frame.metadata_bands(w)
    SampleLogs(w).insert("is_frame_skipping", 0)
    correct_frame.log_band_structure(
        w, BandsTuple(sans_wavelength.Wband(1.5, 2.42), None)
    )
    bands = correct_frame.metadata_bands(w)
    assert bands.lead.min, bands.lead.max == approx(1.5, 2.42)
    SampleLogs(w).insert("is_frame_skipping", 1)
    with pytest.raises(
        RuntimeError, match="Bands from the skipped pulse missing in the logs"
    ):
        correct_frame.metadata_bands(w)
    correct_frame.log_band_structure(
        w,
        BandsTuple(
            sans_wavelength.Wband(1.5, 2.42), sans_wavelength.Wband(6.07, 10.01)
        ),
    )
    bands = correct_frame.metadata_bands(w)
    assert bands.lead.min, bands.lead.max == approx(1.5, 2.42)
    assert bands.skip.min, bands.skip.max == approx(6.07, 10.01)


def test_correct_emission_time_60Hz():
    # excepted wavelengths
    expected_wl = np.arange(1.05, 5.30, 0.1)
    starting_tof = [
        4184.43261874537,
        4585.6620961358,
        4985.12109583935,
        5383.33670855604,
        5780.65753233583,
        6177.18810337877,
        6572.86455843483,
        6967.671528804,
        7362.0002659363,
        7757.14799863174,
        8146.18938057224,
        8537.85660707455,
        8929.51955517686,
        9321.17826927916,
        9712.83279378146,
        10104.4831730838,
        10496.1294515861,
        10887.7716736884,
        11279.4098837907,
        11671.044126293,
        12062.6744455953,
        12454.3008860976,
        12845.9234921999,
        13237.5423083022,
        13629.1573788045,
        14020.7687481068,
        14412.3764606091,
        14803.9805607114,
        15195.5810928137,
        15587.178101316,
        15978.7716306183,
        16370.3617251206,
        16761.9484292229,
        17153.5317873253,
        17545.1118438275,
        17936.6886431299,
        18328.2622296322,
        18719.8326477345,
        19111.3999418368,
        19502.9641563391,
        19894.5253356414,
        20286.0835241437,
        20677.638766246,
    ]

    # Make a simple workspace with correct distances and add tofs to it
    w = CreateSampleWorkspace(
        "Event",
        NumBanks=1,
        BankPixelWidth=1,
        NumEvents=0,
        SourceDistanceFromSample=14.1858856536088,
        BankDistanceFromSample=1.3,
    )
    s = w.getSpectrum(0)
    for tof in starting_tof:
        s.addEventQuickly(float(tof), DateAndTime(0))

    # run correct_emission_time on workspace
    correct_frame.correct_emission_time(w)

    # convert the final tofs to wavelength and compare to expected values
    h = 6.62606896e-34
    m = 1.674927211e-27
    z = 15.4858856536088
    assert_allclose(
        w.getSpectrum(0).getTofs() * 10000 * h / (z * m), expected_wl, rtol=1e-4
    )


def test_correct_emission_time_30Hz():
    # excepted wavelengths
    expected_wl = [
        2.55,
        2.65,
        2.75,
        2.85,
        2.95,
        3.05,
        3.15,
        3.25,
        3.35,
        3.45,
        3.55,
        3.65,
        3.75,
        3.85,
        3.95,
        4.05,
        4.15,
        4.25,
        4.35,
        4.45,
        4.55,
        4.65,
        4.75,
        4.85,
        4.95,
        5.05,
        5.15,
        5.25,
        5.35,
        5.45,
        5.55,
        5.65,
        5.75,
        5.85,
        5.95,
        6.05,
        6.15,
        9.75,
        9.85,
        9.95,
        10.05,
        10.15,
        10.25,
        10.35,
        10.45,
        10.55,
        10.65,
        10.75,
        10.85,
        10.95,
        11.05,
        11.15,
        11.25,
        11.35,
        11.45,
        11.55,
        11.65,
        11.75,
        11.85,
        11.95,
        12.05,
        12.15,
        12.25,
        12.35,
        12.45,
        12.55,
        12.65,
        12.75,
        12.85,
        12.95,
        13.05,
        13.15,
        13.25,
        13.35,
        13.45,
    ]
    starting_tof = [
        11814.9960422188,
        12273.7212567657,
        12732.4424149125,
        13191.1595610593,
        13649.8727396061,
        14108.5819949529,
        14567.2873714998,
        15025.9889136466,
        15484.6866657934,
        15943.3806723402,
        16402.070977687,
        16860.7576262338,
        17319.4406623807,
        17778.1201305275,
        18236.7960750743,
        18695.4685404211,
        19154.1375709679,
        19612.8032111147,
        20071.4655052616,
        20530.1244978084,
        20988.7802331552,
        21447.432755702,
        21906.0821098488,
        22364.7283399956,
        22823.3714905425,
        23282.0116058893,
        23740.6487304361,
        24199.2829085829,
        24657.9141847297,
        25116.5426032765,
        25575.1682086234,
        26033.7910451702,
        26492.411157317,
        26951.0285894638,
        27409.6433860106,
        27868.2555913574,
        28326.8652499043,
        44835.4913471896,
        45294.0379873364,
        45752.5837678833,
        46211.1287332301,
        46669.6729277769,
        47128.2163959237,
        47586.7591820705,
        48045.3013306173,
        48503.8428859642,
        48962.383892511,
        49420.9243946578,
        49879.4644368046,
        50338.0040633514,
        50796.5433186982,
        51255.0822472451,
        51713.6208933919,
        52172.1593015387,
        52630.6975160855,
        53089.2355814323,
        53547.7735419791,
        54006.3114421259,
        54464.8493262728,
        54923.3872388196,
        55381.9252241664,
        55840.4633267132,
        56299.00159086,
        56757.5400610068,
        57216.0787815537,
        57674.6177969005,
        58133.1571514473,
        58591.6968895941,
        59050.2370557409,
        59508.7776942877,
        59967.3188496346,
        60425.8605661814,
        60884.4028883282,
        61342.945860475,
        61801.4895270218,
    ]

    # Make a simple workspace with correct distances and add tofs to it
    w = CreateSampleWorkspace(
        "Event",
        NumBanks=1,
        BankPixelWidth=1,
        NumEvents=0,
        SourceDistanceFromSample=14.1395946855299,
        BankDistanceFromSample=4.0,
    )
    s = w.getSpectrum(0)
    for tof in starting_tof:
        s.addEventQuickly(float(tof), DateAndTime(0))

    # run correct_emission_time on workspace
    correct_frame.correct_emission_time(w)

    # convert the final tofs to wavelength and compare to expected values
    h = 6.62606896e-34
    m = 1.674927211e-27
    z = 18.1395946855299
    assert_allclose(
        w.getSpectrum(0).getTofs() * 10000 * h / (z * m), expected_wl, rtol=1e-4
    )


def test_correct_tof_offset():
    # Make a simple workspace with correct distances and add tofs to it
    w = CreateSampleWorkspace("Event", NumBanks=1, BankPixelWidth=1, NumEvents=10)

    starting_tofs = w.getSpectrum(0).getTofs()

    # set the workspace to frame_skipping, the tofs should be unchanged
    SampleLogs(w).insert("is_frame_skipping", 1)
    # run correct_tof_offset on workspace
    correct_frame.correct_tof_offset(w)
    # compare starting and final tofs
    assert_allclose(w.getSpectrum(0).getTofs(), starting_tofs)

    # set the workspace to not frame_skipping, the tofs should be changed by 664.7
    SampleLogs(w).insert("is_frame_skipping", 0)
    # run correct_tof_offset on workspace
    correct_frame.correct_tof_offset(w)
    # compare starting and final tofs with expected difference
    assert_allclose(w.getSpectrum(0).getTofs(), starting_tofs - 664.7)


if __name__ == "__main__":
    pytest.main([__file__])
