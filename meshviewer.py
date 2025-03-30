import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D model
cube_trimesh = trimesh.load('/Users/rjpearsall/Downloads/YellowCube_Geo_LOW.obj')
mesh = pyrender.Mesh.from_trimesh(cube_trimesh)

# create a scene
scene = pyrender.Scene(ambient_light=[0.01, 0.01, 0.01])

# create a camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# cam_pose = np.array([
#     [1.0, 0.0, 0.0, 23.72112893],
#     [0.0, 1.0, 0.0, 37.55225739],
#     [0.0, 0.0, 1.0, 97.19710764],
#     [0.0, 0.0, 0.0, 1.0]
# ])
cam_pose = np.array([
    [1.0, 0.0, 0.5, 25],
    [0.0, 1.0, 0.0, 15.55225739],
    [0.0, 0.0, 1.0, 97.19710764],
    [0.0, 0.0, 0.0, 1.0]
])
# Create a light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

# Add elements to scene
scene.add(mesh)
scene.add(camera,pose=cam_pose)
scene.add(light, pose=cam_pose)

# scene viewer
# viewer = pyrender.Viewer(scene, use_raymond_lighting=False,viewport_size=(800, 600))

# Create render object
r = pyrender.OffscreenRenderer(800, 800)

# render the scene
color, depth = r.render(scene)

# display the render
plt.imshow(color)
plt.show()







