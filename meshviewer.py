import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt




def light_position(zenith_deg, azimuth_deg, radius=10.0):

    # Convert degrees to radians
    zenith = np.radians(zenith_deg)
    azimuth = np.radians(azimuth_deg)

    # Spherical to Cartesian
    x = radius * np.sin(zenith) * np.cos(azimuth)
    y = radius * np.cos(zenith)
    z = radius * np.sin(zenith) * np.sin(azimuth)

    # Set light's location in 3D space
    eye = np.array([x, y, z])
    # Set the light to always point at the origin
    target = np.array([0.0, 0.0, 0.0])
    # Set the light in the upright direction
    up = np.array([0.0, 1.0, 0.0])

    # forward direction
    z_axis = (target - eye)
    z_axis /= np.linalg.norm(z_axis)

    # right direction
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # corrected up direction
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[0:3, 0] = x_axis
    pose[0:3, 1] = y_axis
    pose[0:3, 2] = z_axis
    pose[0:3, 3] = eye

    return pose


if __name__ == '__main__':


    # Load your 3D model
    cube_trimesh = trimesh.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Computer Vision/Final Project/3D Models/YellowCube/YellowCube_Geo_LOW.obj')
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
    # cam_pose = np.array([
    #     [1.0, 0.0, 0.5, 25],
    #     [0.0, 1.0, 0.0, 15.55225739],
    #     [0.0, 0.0, 1.0, 97.19710764],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])

    cam_pose = np.array([
        [1.0, 0.0, 0.5, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 1.0, 100],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Create a light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    # light rotation
    angle = -np.pi / 2  # -90 degrees
    zenith = np.radians(-100)
    rotation = np.array([
        [1, 0, 0, 0],
        [0, np.cos(zenith), -np.sin(zenith), 0],
        [0, np.sin(zenith), np.cos(zenith), 0],
        [0, 0, 0, 1]
    ])

    rotation = light_position(45,180)

    # Add elements to scene
    scene.add(mesh)
    scene.add(camera,pose=cam_pose)
    scene.add(light, pose=rotation)

    # scene viewer
    # viewer = pyrender.Viewer(scene, use_raymond_lighting=False,viewport_size=(800, 600))

    # Create render object
    r = pyrender.OffscreenRenderer(800, 800)

    # render the scene
    color, depth = r.render(scene)

    # display the render
    plt.imshow(color)
    plt.show()







