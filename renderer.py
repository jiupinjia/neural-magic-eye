import trimesh
import pyrender
import numpy as np
import cv2
import utils

class Renderer():

    def __init__(self, viewport_height, viewport_width, box_scale, yfov):

        self.box_scale = box_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.yfov = yfov

        self.scene = pyrender.Scene()

        self.r = pyrender.OffscreenRenderer(self.viewport_width, self.viewport_height)

    def create_camera(self):
        self.cam = pyrender.PerspectiveCamera(
            yfov=self.yfov / 180. * np.pi,
            aspectRatio=self.viewport_width / self.viewport_height)

    def load_obj_model(self, fname):

        self.obj_nodes = []

        data = trimesh.load(fname, force='scene')
        scenetmp = pyrender.Scene.from_trimesh_scene(data)
        self.base_M = self.create_base_correct_matrix(
            scene=scenetmp, mesh_rotate=[0, 0, 0])

        for x in scenetmp.meshes:
            self.obj_nodes.extend([pyrender.Node(mesh=x, matrix=np.eye(4))])

        for j in range(len(self.obj_nodes)):
            self.scene.add_node(self.obj_nodes[j])

    def clear_scene(self):
        self.scene.clear()

    def _scale_matrix(self, sx=1.0, sy=1.0, sz=1.0):

        ScaleMatrix = np.eye(4)
        ScaleMatrix[0, 0] = sx  # scale on x
        ScaleMatrix[1, 1] = sy  # scale on y
        ScaleMatrix[2, 2] = sz  # scale on z

        return ScaleMatrix

    def _rotation_matrix(self, rx=0., ry=0., rz=0.):

        # input should be degree (e.g., 0, 90, 180)

        # degree to radians
        rx = rx * np.pi / 180.
        ry = ry * np.pi / 180.
        rz = rz * np.pi / 180.

        Rx = np.eye(4)
        Rx[1, 1] = np.cos(rx)
        Rx[1, 2] = -np.sin(rx)
        Rx[2, 1] = np.sin(rx)
        Rx[2, 2] = np.cos(rx)

        Ry = np.eye(4)
        Ry[0, 0] = np.cos(ry)
        Ry[0, 2] = np.sin(ry)
        Ry[2, 0] = -np.sin(ry)
        Ry[2, 2] = np.cos(ry)

        Rz = np.eye(4)
        Rz[0, 0] = np.cos(rz)
        Rz[0, 1] = -np.sin(rz)
        Rz[1, 0] = np.sin(rz)
        Rz[1, 1] = np.cos(rz)

        # RZ * RY * RX
        RotationMatrix = np.mat(Rz) * np.mat(Ry) * np.mat(Rx)

        return np.array(RotationMatrix)

    def _translation_matrix(self, tx=0., ty=0., tz=0.):

        TranslationMatrix = np.eye(4)
        TranslationMatrix[0, -1] = tx
        TranslationMatrix[1, -1] = ty
        TranslationMatrix[2, -1] = tz

        return TranslationMatrix

    def create_pose_matrix(self, tx=0., ty=0., tz=0.,
                           rx=0., ry=0., rz=0.,
                           sx=1.0, sy=1.0, sz=1.0,
                           base_correction=np.eye(4)):

        # Scale matrix
        ScaleMatrix = self._scale_matrix(sx, sy, sz)

        # Rotation matrix
        RotationMatrix = self._rotation_matrix(rx, ry, rz)

        # Translation matrix
        TranslationMatrix = self._translation_matrix(tx, ty, tz)

        # TranslationMatrix * RotationMatrix * ScaleMatrix
        PoseMatrix = np.mat(TranslationMatrix) \
                     * np.mat(RotationMatrix) \
                     * np.mat(ScaleMatrix) \
                     * np.mat(base_correction)

        return np.array(PoseMatrix)

    def create_base_correct_matrix(self, scene, mesh_rotate):

        x_min, x_max = scene.bounds[0, 0], scene.bounds[1, 0]
        y_min, y_max = scene.bounds[0, 1], scene.bounds[1, 1]
        z_min, z_max = scene.bounds[0, 2], scene.bounds[1, 2]

        tx = -(x_max + x_min) / 2.
        ty = -(y_min*0.9)
        tz = -(z_max + z_min) / 2.
        CentralizeMatrix = self.create_pose_matrix(tx=tx, ty=ty, tz=tz)

        obj_scale = ((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)**0.5
        s = self.box_scale / obj_scale
        ScaleMatrix = self.create_pose_matrix(sx=s, sy=s, sz=s)

        RotationMatrix = self.create_pose_matrix(
            rx=mesh_rotate[0], ry=mesh_rotate[1], rz=mesh_rotate[2])

        BaseMatrix = np.mat(RotationMatrix) \
                     * np.mat(ScaleMatrix) \
                     * np.mat(CentralizeMatrix)

        return np.array(BaseMatrix)

    def _mesh_scale(self, mesh):

        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        z = mesh.vertices[:, 2]

        x_scale = np.max(x) - np.min(x)
        y_scale = np.max(y) - np.min(y)
        z_scale = np.max(z) - np.min(z)

        scale = (x_scale ** 2 + y_scale ** 2 + z_scale ** 2) ** 0.5

        return x_scale, y_scale, z_scale, scale

    def set_cam_pose(self, tx=0., ty=0., tz=0., rx=0., ry=0., rz=0., sx=1.0, sy=1.0, sz=1.0):
        pose_matrix = self.create_pose_matrix(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz,
                                                  sx=sx, sy=sy, sz=sz, base_correction=np.eye(4))
        self.cam_node = pyrender.Node(camera=self.cam, matrix=pose_matrix)
        self.scene.add_node(self.cam_node)

    def set_obj_pose(self, tx=0., ty=0., tz=0., rx=0., ry=0., rz=0., sx=1.0, sy=1.0, sz=1.0):
        pose_matrix = self.create_pose_matrix(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz,
                                              sx=sx, sy=sy, sz=sz, base_correction=self.base_M)
        # for each sub-mesh in this obj
        for j in range(len(self.obj_nodes)):
            self.scene.set_pose(self.obj_nodes[j], pose_matrix)

    def render_depthmap(self, d_min=0, d_max=1.0, smooth=False):
    
        flags = pyrender.RenderFlags.DEPTH_ONLY
        depth_map = self.r.render(self.scene, flags=flags)
        mask = (depth_map > 0).astype(np.float32)
        depth_map = utils.normalize(depth_map.max() - depth_map, mask=mask)
        depth_map = (d_max - d_min) * depth_map + d_min
        depth_map = depth_map * mask
        if smooth:
            depth_map = cv2.GaussianBlur(depth_map, ksize=(5, 5), sigmaX=0, sigmaY=0)
            
        return depth_map

