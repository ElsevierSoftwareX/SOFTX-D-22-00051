import numpy as np

__all__ = [
    "stitch_profiles",
]


def stitch_profiles(profiles, overlaps, target_profile_index=0):
    r"""
    Stitch together a sequence of intensity profiles with overlapping domains, returning a single encompassing profile.

    **drtsans objects used**:
    ~drtsans.dataobjects.IQmod
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/dataobjects.py>

    Parameters
    ----------
    profiles: list
        A list of  ~drtsans.dataobjects.IQmod objects, ordered with increasing Q-values
    overlaps: list
        A list of overlap regions in the shape (start_1, end_1, start_2, end_2, start_3, end_3,...).
    target_profile_index: int
        Index of the ``profiles`` list indicating the target profile, that is, the profile defining the final scaling.

    Returns
    -------
    ~drtsans.dataobjects.IQmod
    """
    # Guard clause to verify the profiles are ordered with increasing Q-values
    first_q_values = np.array(
        [profile.mod_q[0] for profile in profiles]
    )  # collect first Q-value for each profile
    if np.all(np.diff(first_q_values) > 0) is False:
        raise ValueError("The profiles are not ordered with increasing Q-values")

    # Guard clause to validate that the number of overlap boundaries is congruent with the number of intensity profiles
    if len(overlaps) != 2 * (len(profiles) - 1):
        raise ValueError(
            "The number of overlaps is not appropriate to the number of intensity profiles"
        )

    # Pair the overlaps into (start_q, end_q) pairs
    overlaps = [overlaps[i : i + 2] for i in range(0, len(overlaps), 2)]

    def scaling(target, to_target, starting_q, ending_q):
        r"""Utility function to find the scaling factor bringing the to_target profile to the target profile scaling"""
        # Find the data points of the "target" profile in the overlap region
        indexes_in_overlap = (
            (target.mod_q > starting_q)
            & (target.mod_q < ending_q)
            & np.isfinite(target.intensity)
        )
        q_values_in_overlap = target.mod_q[indexes_in_overlap]
        # Interpolate the "to_target" profile intensities at the previously found Q values
        good_values = np.isfinite(to_target.intensity)
        to_target_interpolated = np.interp(
            q_values_in_overlap,
            to_target.mod_q[good_values],
            to_target.intensity[good_values],
        )
        scale = sum(target_profile.intensity[indexes_in_overlap]) / \
            sum(to_target_interpolated)
        if scale <= 0:
            raise ValueError(
                f"Scale number: {scale}. The scaling number for stitching cannot be negative. "
                + "Please check the stitching range or profile pattern")
        else:
            return scale

    # We begin stitching to the target profile the neighboring profile with lower Q-values, then proceed until we
    # run out of profiles with lower Q-values than the target profile
    target_profile = profiles[target_profile_index]
    current_index = target_profile_index - 1
    while current_index >= 0:
        to_target_profile = profiles[current_index]
        start_q, end_q = overlaps[current_index]

        # Rescale the "to_target" profile to match the scaling of the target profile
        scale = scaling(target_profile, to_target_profile, start_q, end_q)
        to_target_profile = to_target_profile * scale
        print(
            f"Stitching profile {current_index} to profile {target_profile_index}. Scale factor is {scale:.3e}"
        )

        # Discard extrema points
        to_target_profile = to_target_profile.extract(
            to_target_profile.mod_q < end_q
        )  # keep data with Q < end_q
        target_profile = target_profile.extract(
            target_profile.mod_q > start_q
        )  # keep data with Q > start_q

        # Stitch by concatenation followed by sorting, save the result into a new target profile
        target_profile = to_target_profile.concatenate(
            target_profile
        )  # just put one profile after the other
        target_profile = target_profile.sort()  # sort data points by increasing Q

        # Move to the next to-target profile
        current_index = current_index - 1

    # We continue stitching to the target profile the neighboring profile with higher Q-values, then proceed until we
    # run out of profiles with higher Q-values than the target profile
    current_index = target_profile_index + 1
    while current_index < len(profiles):
        to_target_profile = profiles[current_index]
        start_q, end_q = overlaps[current_index - 1]

        # Rescale the "to_target" profile to match the scaling of the target profile
        scale = scaling(target_profile, to_target_profile, start_q, end_q)
        to_target_profile = to_target_profile * scale
        print(
            f"Stitching profile {current_index} to profile {target_profile_index}. Scale factor is {scale:.3e}"
        )

        # Discard extrema points
        to_target_profile = to_target_profile.extract(
            to_target_profile.mod_q > start_q
        )  # keep data with Q < end_q
        target_profile = target_profile.extract(
            target_profile.mod_q < end_q
        )  # keep data with Q > start_q

        # Stitch by concatenation followed by sorting, save the result into a new target profile
        target_profile = target_profile.concatenate(
            to_target_profile
        )  # just put one profile after the other
        target_profile = target_profile.sort()  # sort data points by increasing Q

        # Move to the next to-target profile
        current_index = current_index + 1

    return target_profile
