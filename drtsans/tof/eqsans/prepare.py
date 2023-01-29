def prepare_beam_center():
    r"""Recipe for the beam center file prior to finding the center coords."""
    pass


def prepare_dark_current():
    r"""Recipe to normalize the dark current"""
    pass


def prepare_sensitivity():
    r"""Recipe to generate a sensitivity file from a flood run"""
    pass


def prepare_scattering():
    r"""Prepare a run containing an element capable of scattering neutrons"""
    pass


def prepare_sample_scattering():
    r"""Prepare a run containing a sample"""
    return prepare_scattering()


def prepare_background_scattering():
    r"""Prepare a run containing an empty sample holder, or a background
    material"""
    return prepare_scattering()


def prepare_transmission():
    r"""
    Prepare a run containing an element capable of scattering neutrons
    attenuator plus an attenuator
    """
    pass


def prepare_sample_transmission():
    r"""Prepare a run containing a sample plus an attenuator"""
    return prepare_transmission()


def prepare_background_transmission():
    r"""
    Prepare a run containing an empty sample holder or a background
    material, plus an attenuator
    """
    return prepare_transmission()
