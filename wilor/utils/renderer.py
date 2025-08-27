import os
# if 'PYOPENGL_PLATFORM' not in os.environ:
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'

import platform
if 'PYOPENGL_PLATFORM' not in os.environ:
    if platform.system() == "Windows":
        os.environ["PYOPENGL_PLATFORM"] = "pyglet"
    else:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional
import open3d as o3d
 

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:,[0,2,1]]

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()

        # renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
        #                                       viewport_height=image.shape[0],
        #                                       point_size=1.0)
        # self.scene = pyrender.Scene()
        # renderer = pyrender.Viewer(self.scene, use_raymond_lighting=True)

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere())
        scene.add(mesh)

        # 用 OffscreenRenderer 可能失败，所以用 Viewer 替代保存
        v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

        # 保存一帧截图
        color, depth = v.render_lock.read()
        from PIL import Image
        Image.fromarray(color).save("output.png")

        v.close_external()


        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        # renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
        #                                       viewport_height=render_res[1],
        #                                       point_size=1.0)
        # self.scene = pyrender.Scene()
        # renderer = pyrender.Viewer(self.scene, use_raymond_lighting=True)

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere())
        scene.add(mesh)

        # 用 OffscreenRenderer 可能失败，所以用 Viewer 替代保存
        v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

        # 保存一帧截图
        color, depth = v.render_lock.read()
        from PIL import Image
        Image.fromarray(color).save("output.png")

        v.close_external()


        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        # renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
        #                                       viewport_height=render_res[1],
        #                                       point_size=1.0)
        # self.scene = pyrender.Scene()
        # renderer = pyrender.Viewer(self.scene, use_raymond_lighting=True)

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere())
        scene.add(mesh)

        # 用 OffscreenRenderer 可能失败，所以用 Viewer 替代保存
        v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

        # 保存一帧截图
        color, depth = v.render_lock.read()
        from PIL import Image
        Image.fromarray(color).save("output.png")

        v.close_external()


        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)


import numpy as np


def trimesh_to_o3d(vertices, faces, vertex_colors=None):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.compute_vertex_normals()
    if vertex_colors is not None:
        # O3D 顶点颜色为 Nx3（RGB，0~1）
        if vertex_colors.shape[1] == 4:
            vertex_colors = vertex_colors[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(vertex_colors[:, :3], 0, 1).astype(np.float64)
        )
    return mesh

def make_material(base_color=(1.0, 1.0, 0.9, 1.0), metallic=0.0, roughness=0.6, transparent=False):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency" if transparent or (len(base_color) == 4 and base_color[3] < 1.0) else "defaultLit"
    rgba = list(base_color) if len(base_color) == 4 else list(base_color) + [1.0]
    mat.base_color = rgba
    mat.base_roughness = float(roughness)
    mat.base_reflectance = 0.5
    mat.base_metallic = float(metallic)
    mat.albedo_img = None
    return mat

def rot_x_180(mesh_o3d: o3d.geometry.TriangleMesh):
    # 等价于你在 pyrender 里：绕 X 轴旋转 180°
    R = mesh_o3d.get_rotation_matrix_from_xyz([np.pi, 0, 0])
    mesh_o3d.rotate(R, center=(0, 0, 0))


import open3d as o3d
class RendererO3D:
    def __init__(self, cfg, faces: np.ndarray):
        self.cfg = cfg
        self.focal_length = float(cfg.EXTRA.FOCAL_LENGTH)
        self.img_res = int(cfg.MODEL.IMAGE_SIZE)
        # 手部补面同你原始逻辑
        faces_new = np.array([[92, 38, 234],[234, 38, 239],[38, 122, 239],[239, 122, 279],
                              [122, 118, 279],[279, 118, 215],[118, 117, 215],[215, 117, 214],
                              [117, 119, 214],[214, 119, 121],[119, 120, 121],[121, 120, 78],
                              [120, 108, 78],[78, 108, 79]], dtype=np.int32)
        self.faces = np.concatenate([faces.astype(np.int32), faces_new], axis=0)
        self.faces_left = self.faces[:, [0, 2, 1]]

    # —— 等价于你 vertices_to_trimesh 的颜色/左右手/旋转处理 ——
    def vertices_to_o3d_mesh(
        self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
        rot_axis=(1, 0, 0), rot_angle=0, is_right=1
    ):
        vtx = vertices.copy()
        fcs = self.faces if is_right else self.faces_left
        # 颜色：顶点全填充
        vertex_colors = np.tile(np.array(list(mesh_base_color)[:3], dtype=np.float32), (vtx.shape[0], 1))
        mesh = trimesh_to_o3d(vtx + camera_translation, fcs, vertex_colors)

        # 自定义旋转
        axis = np.array(rot_axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(rot_angle))
        mesh.rotate(R, center=(0, 0, 0))

        # X 轴 180° 翻转与原 pyrender 逻辑一致
        rot_x_180(mesh)
        return mesh

    # —— 单人手 RGBA 渲染（替代 render_rgba） ——
    def render_rgba(
        self,
        vertices: np.ndarray,
        cam_t=None,     # np.array([tx, ty, tz]) 相机中心(世界系)
        rot=None,       # 保留接口，不用也可
        rot_axis=(1, 0, 0),
        rot_angle=0,
        camera_z=3,
        mesh_base_color=(1.0, 1.0, 0.9, 1.0),
        scene_bg_color=(0, 0, 0, 0),
        render_res=(256, 256),
        focal_length=None,
        is_right=1,
    ):
        W, H = int(render_res[0]), int(render_res[1])
        fx = float(focal_length) if focal_length is not None else float(self.focal_length)
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        # 准备渲染器
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([*scene_bg_color])  # RGBA
        # （可选）环境光/间接光，增强整体亮度与柔和度
        renderer.scene.scene.set_indirect_light_intensity(50000)  # 数值可按需微调
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light([0, -1, 1], [1, 1, 1], 75000, True)

        # 相机：内参
        znear, zfar = 0.01, 1e4
        renderer.scene.camera.set_projection(fx, fy, cx, cy, W, H, znear, zfar)

        # 几何体
        mesh = self.vertices_to_o3d_mesh(
            vertices,
            np.zeros(3, dtype=np.float64),
            mesh_base_color=mesh_base_color,
            rot_axis=rot_axis, rot_angle=rot_angle, is_right=is_right
        )
        mat = make_material(base_color=mesh_base_color, transparent=(len(mesh_base_color) == 4 and mesh_base_color[3] < 1.0))
        renderer.scene.add_geometry("mesh", mesh, mat)

        # 相机：外参（眼睛位置 & LookAt）
        if cam_t is not None:
            eye = np.array([ -cam_t[0],  cam_t[1], cam_t[2] ], dtype=np.float64)  # 你原代码里有 x 取负
        else:
            eye = np.array([0, 0, camera_z * fx / H], dtype=np.float64)
        center = np.array([0, 0, 0], dtype=np.float64)
        up = np.array([0, -1, 0], dtype=np.float64)  # 和 X 轴 180° 一起，得到与 pyrender 相同的成像方向
        renderer.scene.camera.look_at(center, eye, up)

        # 渲染
        img = renderer.render_to_image()  # o3d.geometry.Image，uint8 RGBA
        rgba = np.asarray(img)  # HxWx4
        rgba = rgba.astype(np.float32) / 255.0
        return rgba

    # —— 多人手 RGBA（替代 render_rgba_multiple） ——
    def render_rgba_multiple(
        self,
        vertices,                 # List[np.ndarray]，每个 (V,3)
        cam_t,                    # List[np.ndarray]，每个 (3,)
        rot_axis=(1, 0, 0),
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9, 1.0),
        scene_bg_color=(0, 0, 0, 0),
        render_res=(256, 256),
        focal_length=None,
        is_right=None,
    ):
        import open3d as o3d
        import numpy as np

        W, H = int(render_res[0]), int(render_res[1])
        fx = float(focal_length) if focal_length is not None else float(self.focal_length)
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([*scene_bg_color])  # 3或4通道都可
        renderer.scene.scene.set_indirect_light_intensity(50000)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light([0, -1, 1], [1, 1, 1], 75000, True)

        renderer.scene.camera.set_projection(fx, fy, cx, cy, W, H, 0.01, 1e4)

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        for i, (vtx, t, right_flag) in enumerate(zip(vertices, cam_t, is_right)):
            t = np.asarray(t, dtype=np.float64)
            # 每个 mesh 按对应 cam_t 平移（复现你原 pyrender 的做法）
            mesh = self.vertices_to_o3d_mesh(
                vtx,
                t,
                mesh_base_color=mesh_base_color[:3],
                rot_axis=rot_axis,
                rot_angle=rot_angle,
                is_right=right_flag
            )
            mat = make_material(base_color=mesh_base_color)
            renderer.scene.add_geometry(f"mesh_{i}", mesh, mat)

        # 相机放在原点前方，朝原点看（与原多手渲染一致）
        eye = np.array([0, 0, fx / H * 3], dtype=np.float64)
        center = np.array([0, 0, 0], dtype=np.float64)
        up = np.array([0, -1, 0], dtype=np.float64)
        renderer.scene.camera.look_at(center, eye, up)

        img = renderer.render_to_image()
        rgba = np.asarray(img).astype(np.float32) / 255.0
        return rgba
    
    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                        rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        import numpy as np
        import trimesh

        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0], dtype=np.float32)
        faces = self.faces if is_right else self.faces_left
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, faces.copy(),
                            vertex_colors=vertex_colors)

        # 自定义旋转
        R1 = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(R1)
        # 与原逻辑一致：绕 X 轴 180°
        R2 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(R2)
        return mesh
