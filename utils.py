import numpy as np
from numba import njit, prange

@njit
def numba_percentile(data, q):
    sorted_data = np.sort(data)
    rank = (q / 100.0) * (len(sorted_data) - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return sorted_data[lower]
    else:
        weight = rank - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

@njit
def unit_vector(v):
    norm = np.sqrt(np.sum(v**2))
    if norm == 0:
        return np.zeros(3, dtype=np.float64)
    return (v / norm).astype(np.float64) 

@njit
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    dot_product = max(min(dot_product, 1.0), -1.0)
    return np.arccos(dot_product)

@njit
def angle_change(ii, normals):
    angle_p = np.empty(len(ii))
    for n in prange(len(ii)):
        angles = np.empty(len(ii[n]))
        for j in prange(len(ii[n])):
            angles[j] = angle_between(normals[n], normals[ii[n][j]])
        angle_p[n] = numba_percentile(angles, 95)
    return angle_p