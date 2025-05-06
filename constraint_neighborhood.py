from utils import *
from scipy.spatial import cKDTree

@njit()
def constraint_region_growing(seeds, connections, points, radius, normals, orientation_tolerance):
    """
    Parameters:
    - seeds: list of seed points to compute the neighborhoods for
    - connections: list of lists of indices of k nearest neighbors for each point
    - points: xyz-array of point coordinates
    - radius: maximum distance of the neighborhood
    - normals: array of point normals
    - orientation_tolerance: orientation tolerance for each point

    Returns:
    - neighborhoods: list of neighborhoods for each seed point
    """
    neighborhoods = []
    radius_squared = radius ** 2  # Compare squared distances to avoid square root calculation

    for point in seeds:
        neighborhood = []
        nn_count = 0
        queue = np.empty(len(connections), dtype=np.int32)
        queue[0] = point
        start = 0
        end = 1
        touch = np.zeros(len(points), dtype="bool")
        touch[point] = True

        p_tolerance = orientation_tolerance[point]
        p_normal = normals[point]
        p_xyz = points[point]

        while start < end:
            i = queue[start]
            start += 1

            for j in connections[i]:
                if not touch[j]:
                    dx = points[j, 0] - p_xyz[0]
                    dy = points[j, 1] - p_xyz[1]
                    dz = points[j, 2] - p_xyz[2]
                    dist_squared = dx * dx + dy * dy + dz * dz

                    if dist_squared <= radius_squared and angle_between(p_normal, normals[j]) <= p_tolerance:
                        queue[end] = j
                        end += 1
                        neighborhood.append(j)
                        nn_count += 1
                        touch[j] = True

        neighborhoods.append(neighborhood)

    return neighborhoods