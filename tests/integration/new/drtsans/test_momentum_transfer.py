import numpy as np
import pytest

r"""
Hyperlinks to mantid algorithms
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
LoadInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadInstrument-v1.html>
"""
from mantid.simpleapi import CreateWorkspace, LoadInstrument

r"""
Hyperlinks to drtsans functions
convert_to_q <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/momentum_transfer.py>
namedtuplefy, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
"""  # noqa: E501
from drtsans.momentum_transfer import convert_to_q
from drtsans.settings import namedtuplefy, unique_workspace_dundername


# Instrument definition file for two tubes, each tube contains two pixels.
# Pixels are cylinders of diameter 0.02m and height 0.02m
coarse_instrument = r"""
<?xml version="1.0" encoding="UTF-8"?>
<instrument name="GenericSANS" valid-from="1900-01-31 23:59:59" valid-to="2100-12-31 23:59:59"
 last-modified="2019-07-12 00:00:00">
    <!--DEFAULTS-->
    <defaults>
        <length unit="metre"/>  <angle unit="degree"/> <reference-frame> <along-beam axis="z"/>
        <pointing-up axis="y"/> <handedness val="right"/> <theta-sign axis="x"/>  </reference-frame>
    </defaults>

    <!--SOURCE-->
    <component type="moderator">
        <location z="-11.0"/>
    </component>
    <type name="moderator" is="Source"/>

    <!--SAMPLE-->
    <component type="sample-position">
        <location y="0.0" x="0.0" z="0.0"/>
    </component>
    <type name="sample-position" is="SamplePos"/>

    <!---->
    <!--TYPE: PIXEL FOR STANDARD PIXEL TUBE-->
    <!---->
    <type name="pixel" is="detector">
        <cylinder id="cyl-approx">
            <centre-of-bottom-base r="0.0" t="0.0" p="0.0"/> <axis x="0.00000" y="1.00000" z="0.00000"/>
            <radius val="0.01"/> <height val="0.02"/>
        </cylinder>
      <algebra val="cyl-approx"/>
    </type>

    <!---->
    <!--TYPE: STANDARD PIXEL TUBE-->
    <!---->
    <type outline="yes" name="tube">
    <properties/>
    <component type="pixel">
        <location name="pixel0" y="-0.01000000"/>
        <location name="pixel1" y="0.01000000"/>
    </component>
  </type>

    <!---->
    <!--TYPE: N-PACK-->
    <!---->
      <type name="n_pack">
    <properties/>
    <component type="tube">
        <location name="tube0" x="-0.01000000"/>
        <location name="tube1" x="0.01000000"/>
    </component>
    </type>

    <!---->
    <!--COMPONENT: N-PACK-->
    <!---->
    <component type="n_pack" idlist="n_panel_ids" name="detector1">
        <location x="0.0" y="0.0" z="1.0" rot="180.0" axis-x="0" axis-y="1" axis-z="0">
        </location>
    </component>

    <!--DETECTOR IDs-->
    <idlist idname="n_panel_ids">
        <id start="0" end="3"/>
    </idlist>

    <parameter name="x-pixel-size">
        <value val="20.0"/>
    </parameter>

    <parameter name="y-pixel-size">
        <value val="20.0"/>
    </parameter>
</instrument>
"""

# Instrument definition file for four tubes, each tube contains four pixels.
# Pixels are cylinders of diameter 0.01m and height 0.01m
fine_instrument = r"""
<?xml version="1.0" encoding="UTF-8"?>
<instrument name="GenericSANS" valid-from="1900-01-31 23:59:59" valid-to="2100-12-31 23:59:59"
 last-modified="2019-07-12 00:00:00">
    <!--DEFAULTS-->
    <defaults>
        <length unit="metre"/>  <angle unit="degree"/> <reference-frame> <along-beam axis="z"/>
        <pointing-up axis="y"/> <handedness val="right"/> <theta-sign axis="x"/>  </reference-frame>
    </defaults>

    <!--SOURCE-->
    <component type="moderator">
        <location z="-11.0"/>
    </component>
    <type name="moderator" is="Source"/>

    <!--SAMPLE-->
    <component type="sample-position">
        <location y="0.0" x="0.0" z="0.0"/>
    </component>
    <type name="sample-position" is="SamplePos"/>

    <!---->
    <!--TYPE: PIXEL FOR STANDARD PIXEL TUBE-->
    <!---->
    <type name="pixel" is="detector">
        <cylinder id="cyl-approx">
            <centre-of-bottom-base r="0.0" t="0.0" p="0.0"/> <axis x="0.00000" y="1.00000" z="0.00000"/>
            <radius val="0.005"/> <height val="0.01"/>
        </cylinder>
      <algebra val="cyl-approx"/>
    </type>

    <!---->
    <!--TYPE: STANDARD PIXEL TUBE-->
    <!---->
    <type outline="yes" name="tube">
    <properties/>
    <component type="pixel">
        <location name="pixel0" y="-0.01500000"/>
        <location name="pixel1" y="-0.00500000"/>
        <location name="pixel2" y="0.0050000"/>
        <location name="pixel3" y="0.01500000"/>
    </component>
  </type>

    <!---->
    <!--TYPE: N-PACK-->
    <!---->
      <type name="n_pack">
    <properties/>
    <component type="tube">
        <location name="tube0" x="-0.01500000"/>
        <location name="tube1" x="-0.00500000"/>
        <location name="tube2" x="0.00500000"/>
        <location name="tube3" x="0.01500000"/>
    </component>
    </type>

    <!---->
    <!--COMPONENT: N-PACK-->
    <!---->
    <component type="n_pack" idlist="n_panel_ids" name="detector1">
        <location x="0.0" y="0.0" z="1.0" rot="180.0" axis-x="0" axis-y="1" axis-z="0">
        </location>
    </component>

    <!--DETECTOR IDs-->
    <idlist idname="n_panel_ids">
        <id start="0" end="15"/>
    </idlist>

    <parameter name="x-pixel-size">
        <value val="10.0"/>
    </parameter>

    <parameter name="y-pixel-size">
        <value val="10.0"/>
    </parameter>
</instrument>
"""


@pytest.fixture(scope="function")
@namedtuplefy
def data_subpixels_convert_to_q():
    r"""Data to be used in test_subpixels_convert_to_q()"""
    return dict(
        wavelength_bin_boundaries=[2.0, 2.1, 2.2],
        coarse_intensities=[
            [1.0, 1.1],
            [2.0, 2.1],  # first tube
            [3.0, 3.1],
            [4.0, 4.1],
        ],  # second tube
        fine_intensities=[
            [[1.0, 1.1], [1.0, 1.1], [2.0, 2.1], [2.0, 2.1]],
            [[1.0, 1.1], [1.0, 1.1], [2.0, 2.1], [2.0, 2.1]],
            [[3.0, 3.1], [3.0, 3.1], [4.0, 4.1], [4.0, 4.1]],
            [[3.0, 3.1], [3.0, 3.1], [4.0, 4.1], [4.0, 4.1]],
        ],
        n_h=2,  # subpixels in X-direction
        n_v=2,  # subpixels in vertical (Y) direction
    )


def test_subpixels_convert_to_q(data_subpixels_convert_to_q):
    r"""
    Compare Q values from the coarse instrument where we divide each pixel into a 2x2 = 4 subpixels with
    Q values from the fine instrument.
    The number of subpixels in the coarse instrument is the same as the number of pixels in the fine instrument.
    Furthermore, the location and dimensions of the subpixels in the coarse instrument coincides with the
    location and dimensions of the pixels in the fine instrument. Hence, Q values and intensities between
    corresponding subpixel and pixel should match

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    """
    data = data_subpixels_convert_to_q  # handy shortcut
    # Workspace containing an instrument made of two tubes, and two pixels per tube
    coarse_workspace = CreateWorkspace(
        DataX=data.wavelength_bin_boundaries,
        UnitX="Wavelength",
        DataY=np.array(data.coarse_intensities),
        DataE=np.sqrt(data.coarse_intensities),
        NSpec=4,
        OutputWorkspace=unique_workspace_dundername(),
    )
    LoadInstrument(
        Workspace=coarse_workspace,
        InstrumentXML=coarse_instrument,
        RewriteSpectraMap=True,
        InstrumentName="GenericSANS",
    )
    assert coarse_workspace.extractY().shape == (
        4,
        2,
    )  # 4 pixels, each pixel contains two intensity values
    # Workspace containing an instrument made of four tubes, and four pixels per tube
    fine_workspace = CreateWorkspace(
        DataX=data.wavelength_bin_boundaries,
        UnitX="Wavelength",
        DataY=np.array(data.fine_intensities),
        DataE=np.sqrt(data.fine_intensities),
        NSpec=16,
        OutputWorkspace=unique_workspace_dundername(),
    )
    LoadInstrument(
        Workspace=fine_workspace,
        InstrumentXML=fine_instrument,
        RewriteSpectraMap=True,
        InstrumentName="GenericSANS",
    )
    assert fine_workspace.extractY().shape == (
        16,
        2,
    )  # 16 pixels, each pixel contains two intensity values

    n_bins = len(data.wavelength_bin_boundaries) - 1  # number of wavelength bins
    # Subpixel indexes in the coarse instrument do not correspond to pixel indexes in the fine instrument.
    #         coarse instrument                  fine instrument pixel indexes
    #         subpixel indexes                        -------------------
    #         ----------------                        | 3 | 7 | 11 | 15 |
    #         | 5  7 | 13 15 |                        |-----------------|
    #         | 4  6 | 12 14 |                        | 2 | 6 | 10 | 14 |
    #         |--------------|                        |-----------------|
    #         | 1  3 |  9 11 |                        | 1 | 5 |  9 | 13 |
    #         | 0  2 |  8 10 |                        |-----------------|
    #         ----------------                        | 0 | 4 |  8 | 12 |
    #                                                 -------------------
    # Subpixel indexes are generated as follows:
    # 1. Divide pixel with index 0 into four pixels. The pixel becomes a little detector with two "tubes" and two
    # "pixels per tube":
    #       pixel with index 0       is split into 4 subpixels.   Compare to pixel indexes of the fine insrument
    #       ----------------              ----------------                -------------------
    #       |      |      |               |      |      |                 |   |   |    |    |
    #       |      |      |               |      |      |                 |-----------------|
    #       |-------------|               |-------------|                 |   |   |    |    |
    #       | **** |      |               | 1  3 |      |                 |-----------------|
    #       | **** |      |               | 0  2 |      |                 | 1 | 5 |    |    |
    #       ---------------               ---------------                 |-----------------|
    #                                                                     | 0 | 4 |    |    |
    #                                                                     -------------------
    # 2. Divive pixel with index 1 into four pixels. The index for the firs subpixel of pixel 2 will be "4".
    #
    # We require a permutation of the coarse subpixel indexes in order to compare results from the coarse
    # instrument with subpixels to results from the fine instrument.
    permutation = [
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        10,
        11,
        14,
        15,
    ]  # from coarse index to fine index

    # Test modulus Q between subpixels from the coarse instrument and pixels from the fine instrument
    # "coarse_iq" is an IQmod object, containing values for the intensity, error, Q-modulus, wavelength
    coarse_iq = convert_to_q(
        coarse_workspace, "scalar", n_horizontal=data.n_h, n_vertical=data.n_v
    )
    fine_iq = convert_to_q(
        fine_workspace, "scalar"
    )  # no subpixels here (n_horizontal = n_vertical = 1)
    # check the following quantities are identical between the two scenarios
    for quantity in ("mod_q", "intensity", "wavelength"):
        coarse_values = getattr(coarse_iq, quantity).reshape(-1, n_bins)[permutation]
        fine_values = getattr(fine_iq, quantity).reshape(-1, n_bins)
        assert coarse_values == pytest.approx(fine_values)
    # Errors should be different by a factor of sqrt(data.n_h*data.n_v)
    coarse_values = getattr(coarse_iq, "error").reshape(-1, n_bins)[permutation]
    fine_values = getattr(fine_iq, "error").reshape(-1, n_bins)
    assert coarse_values == pytest.approx(fine_values * np.sqrt(data.n_h * data.n_v))

    # Test azimuthal Q (Qx, Qy) between subpixels and fine pixels
    # "coarse_iq" is an IQazimuthal object, containing values for the intensity, error, Qx, Qy, wavelength
    coarse_iq = convert_to_q(
        coarse_workspace, "azimuthal", n_horizontal=data.n_h, n_vertical=data.n_v
    )
    fine_iq = convert_to_q(fine_workspace, "azimuthal")
    for quantity in ("qx", "qy", "intensity", "wavelength"):
        coarse_values = getattr(coarse_iq, quantity).reshape(-1, n_bins)[permutation]
        fine_values = getattr(fine_iq, quantity).reshape(-1, n_bins)
        assert coarse_values == pytest.approx(fine_values)
    # Errors should be different by a factor of sqrt(data.n_h*data.n_v)
    coarse_values = getattr(coarse_iq, "error").reshape(-1, n_bins)[permutation]
    fine_values = getattr(fine_iq, "error").reshape(-1, n_bins)
    assert coarse_values == pytest.approx(fine_values * np.sqrt(data.n_h * data.n_v))

    # Test crystal Q (Qx, Qy, Qz) between subpixels and fine pixels
    # "coarse_iq" is an IQcrystal object, containing values for the intensity, error, Qx, Qy, Qz, wavelength
    coarse_iq = convert_to_q(
        coarse_workspace, "crystallographic", n_horizontal=data.n_h, n_vertical=data.n_v
    )
    fine_iq = convert_to_q(fine_workspace, "crystallographic")
    for quantity in ("qx", "qy", "qz", "intensity", "wavelength"):
        coarse_values = getattr(coarse_iq, quantity).reshape(-1, n_bins)[permutation]
        fine_values = getattr(fine_iq, quantity).reshape(-1, n_bins)
        assert coarse_values == pytest.approx(fine_values)
    # Errors should be different by a factor of sqrt(data.n_h*data.n_v)
    coarse_values = getattr(coarse_iq, "error").reshape(-1, n_bins)[permutation]
    fine_values = getattr(fine_iq, "error").reshape(-1, n_bins)
    assert coarse_values == pytest.approx(fine_values * np.sqrt(data.n_h * data.n_v))


if __name__ == "__main__":
    pytest.main([__file__])
