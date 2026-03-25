import numpy as np

VALID_OCCLUSION_LEVELS = {"0%", "50%_random", "75%_block"}

def apply_synthetic_occlusion(times, fluxes, errors, passbands, level):
    """
    Apply synthetic data occlusion to simulate real-world telescope disruptions.

    Parameters
    ----------
    times, fluxes, errors, passbands : np.ndarray
        Parallel arrays of observations for a single astronomical object.
    level : str
        One of '0%', '50%_random', or '75%_block'.

    Returns
    -------
    Filtered tuple of (times, fluxes, errors, passbands).
    """
    # FAILSAFE: guard against invalid occlusion level strings passed at runtime
    if level not in VALID_OCCLUSION_LEVELS:
        raise ValueError(
            f"Unknown occlusion level '{level}'. "
            f"Valid options are: {sorted(VALID_OCCLUSION_LEVELS)}"
        )

    seq_len = len(times)

    # FAILSAFE: empty input sequence — return as-is, caller handles empty arrays
    if seq_len == 0:
        return times, fluxes, errors, passbands

    if level == "0%":
        return times, fluxes, errors, passbands

    elif level == "50%_random":
        n_keep = max(1, int(seq_len * 0.5))  # keep at least 1 observation
        keep_indices = np.sort(np.random.choice(seq_len, size=n_keep, replace=False))
        return (times[keep_indices], fluxes[keep_indices],
                errors[keep_indices], passbands[keep_indices])

    elif level == "75%_block":
        # Simulate a 3-month telescope failure (approx 75% of sequence)
        start_idx = np.random.randint(0, max(1, int(seq_len * 0.25)))
        end_idx = min(start_idx + int(seq_len * 0.75), seq_len - 1)

        keep_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, seq_len)])
        # FAILSAFE: ensure at least one element survives extreme masking
        if len(keep_indices) == 0:
            keep_indices = np.array([0])

        return (times[keep_indices], fluxes[keep_indices],
                errors[keep_indices], passbands[keep_indices])

    # Unreachable due to guard above, but kept for static analysis tools
    return times, fluxes, errors, passbands
