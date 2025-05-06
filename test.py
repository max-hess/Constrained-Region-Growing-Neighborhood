from utils import *
from constraint_neighborhood import constraint_region_growing

from scipy.spatial import cKDTree
import pyvista as pv

# example usage

# data
n = 5000
face1 = np.random.uniform(size = (n // 2, 3))
face1[:, 2] *= 0.05
face1[face1[:, 0] > 0.5, 0] += 0.1
face2 = np.random.uniform(size = (n // 2, 3))
face2[:, 0] *= 0.05
points = np.concatenate((face1, face2), axis=0)

# Generate normals
normals_face1 = np.tile([0, 0, 1], (n // 2, 1))
normals_face2 = np.tile([1, 0, 0], (n // 2, 1))
normals = np.concatenate((normals_face1, normals_face2), axis=0)


# parameters
seeds = np.arange(n)
k = 6
radius = 0.4

# calculate orientation tolerance
tree = cKDTree(points)
__, kk = tree.query(points, k=100+1, workers=-1)
kk = kk[:, 1:]
p95 = angle_change(kk, normals)

orientation_tolerance = np.where(p95 < np.pi/4, np.pi/4, p95)
__, kk = tree.query(points, k=k+1, workers=-1)
kk = kk[:, 1:]
constraint_neighbors = constraint_region_growing(seeds, kk, points, radius, normals, orientation_tolerance)

# visualize results
reference_point = 500
color = np.ones(n)
color[reference_point] = 0
color[constraint_neighbors[reference_point]] = 2

# Create the PolyData object
point_cloud = pv.PolyData(points)
point_cloud["scalars"] = color
point_cloud["point_id"] = np.arange(n)

# Callback function for point picking
def if_picked(picker, id):
    color = np.ones(n)
    color[id] = 0
    color[constraint_neighbors[id]] = 2
    point_cloud["scalars"] = color    
    p.update()  

# Set up the plotter and add the mesh
p = pv.Plotter()
p.add_mesh(point_cloud, scalars="scalars", render_points_as_spheres=True, point_size=10, pickable=True)

# Enable point picking with the callback
p.enable_point_picking(callback=if_picked, use_mesh=True)

# Show the plotter
p.show()

