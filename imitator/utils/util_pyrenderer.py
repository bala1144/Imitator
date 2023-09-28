import numpy as np
import pyrender
import trimesh
import cv2
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags

class Facerender:
    def __init__(self, intrinsic=(2035.18464, -2070.36928, 257.55392, 256.546816),
                 img_size=(512, 512)):
        self.image_size = img_size
        self.scene = pyrender.Scene(ambient_light=[.75, .75, .75], bg_color=[0, 0, 0])

        # create camera and light
        self.add_camera(intrinsic)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        self.scene.add(light, pose=np.eye(4))
        self.r = pyrender.OffscreenRenderer(*self.image_size)
        self.mesh_node = None

    def add_camera(self, intrinsic):
        (fx, fy, Cx, Cy) = intrinsic
        camera = pyrender.camera.IntrinsicsCamera(fx, fy, Cx, Cy,
                                                  znear=0.05, zfar=10.0, name=None)

        camera_rotation = np.eye(4)
        camera_rotation[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler(
            'x', 0, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1])
        camera_pose = camera_rotation @ camera_translation
        self.scene.add(camera, pose=camera_pose)

    def add_face(self, vertices, faces, pose=np.eye(4)):
        """
        Input :
         vertices : N x 3
         faces: F x 3
        """
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        tri_mesh = trimesh.Trimesh(vertices, faces)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)
    

    def add_face_v2(self, vertices, faces, pose=np.eye(4)):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        primitive = [pyrender.Primitive(
                    positions=vertices.copy(),
                    indices=faces,
                    material = pyrender.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.8),
                mode=pyrender.GLTF.TRIANGLES)
                    ]
        mesh = pyrender.Mesh(primitives=primitive, is_visible=True)
        self.mesh_node = self.scene.add(mesh, pose=pose)


    def add_vertics(self, vertices, point_radius=0.01, vertex_colour=[0.0, 0.0, 1.0]):
        sm = trimesh.creation.uv_sphere(radius=point_radius)
        sm.visual.vertex_colors = vertex_colour
        tfs = np.tile(np.eye(4), (vertices.shape[0], 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self.mesh_node = self.scene.add(vertices_Render, pose=np.eye(4))

    def add_obj(self, obj_path):

        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        mesh = trimesh.load(obj_path)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.mesh_node = self.scene.add(mesh)

    def render(self):
        flags = RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color

    def live(self):
        pyrender.Viewer(self.scene)