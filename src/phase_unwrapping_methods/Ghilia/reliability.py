import numpy as np

def reliability_weight_map(wrapped_phase, eps=1e-6):
    """
    Compute a reliability/weight map for weighted least squares phase unwrapping.
    Based on Arevalillo-Herráez (2001), also used in scikit-image unwrap.

    Parameters
    ----------
    wrapped_phase : 2D ndarray
        Wrapped phase image in radians.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    weights : 2D ndarray
        Reliability weights (higher = more reliable).
    """
    ny, nx = wrapped_phase.shape
    R = np.zeros_like(wrapped_phase, dtype=float)

    # second differences
    H = np.abs(wrapped_phase[:, :-2] - 2*wrapped_phase[:, 1:-1] + wrapped_phase[:, 2:])
    V = np.abs(wrapped_phase[:-2, :] - 2*wrapped_phase[1:-1, :] + wrapped_phase[2:, :])
    D1 = np.abs(wrapped_phase[:-2, :-2] - 2*wrapped_phase[1:-1, 1:-1] + wrapped_phase[2:, 2:])
    D2 = np.abs(wrapped_phase[:-2, 2:] - 2*wrapped_phase[1:-1, 1:-1] + wrapped_phase[2:, :-2])

    # insert into central region
    R[1:-1, 1:-1] = H[1:-1, :] + V[:, 1:-1] + D1 + D2

    # convert to weights
    W = 1.0 / (R + eps)
    
    return W

def balanced_reliability_weight_map(wrapped_phase, amplitude=None, eps=1e-6, grad_sigma=0.3):
    """
    Reliability weights for phase unwrapping.
    Penalizes noisy or too-flat regions.
    """
    ny, nx = wrapped_phase.shape
    R = np.zeros_like(wrapped_phase, dtype=float)

    H = np.abs(wrapped_phase[:, :-2] - 2*wrapped_phase[:, 1:-1] + wrapped_phase[:, 2:])
    V = np.abs(wrapped_phase[:-2, :] - 2*wrapped_phase[1:-1, :] + wrapped_phase[2:, :])
    D1 = np.abs(wrapped_phase[:-2, :-2] - 2*wrapped_phase[1:-1, 1:-1] + wrapped_phase[2:, 2:])
    D2 = np.abs(wrapped_phase[:-2, 2:] - 2*wrapped_phase[1:-1, 1:-1] + wrapped_phase[2:, :-2])
    R[1:-1, 1:-1] = H[1:-1, :] + V[:, 1:-1] + D1 + D2

    W = 1.0 / (R + eps)

    gy, gx = np.gradient(wrapped_phase)
    grad_mag = np.sqrt(gx**2 + gy**2)
    flat_penalty = 1 - np.exp(-(grad_mag / grad_sigma)**2)
    W *= flat_penalty

    if amplitude is not None:
        A = amplitude / (np.max(amplitude) + eps)
        W *= A

    W = gaussian_filter(W, sigma=1)
    W /= np.max(W) + eps
    W += 1e-3

    return W
