import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import hydra
from utils_infer import Trainer
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from PIL import Image
import time
from visualization_tools import ImageGridViewer, PointCloudImageViewer, PointCloudViewer

def save_point_cloud_to_image(pcd, image_size=(1600, 1200)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_size[0], height=image_size[1])  # 设置窗口大小并不显示

    # 添加点云到可视化器中
    vis.add_geometry(pcd)

    # 获取 ViewControl 对象并设置自定义视角
    view_control = vis.get_view_control()

    # 设置视角参数
    parameters = {
			"boundingbox_max" : [ -1.5977719363706235, 11.519330868353832, 84.127326965332031 ],
			"boundingbox_min" : [ -56.623178268627761, -50.724836932361825, 4.6948032379150391 ],
			"field_of_view" : 60.0,
			"front" : [ 0.36788389015943362, 0.28372788418091033, -0.8855280521244856 ],
			"lookat" : [ -29.110475102499194, -19.602753032003996, 44.411065101623535 ],
			"up" : [ -0.88654778269461509, -0.18027422226025128, -0.42606834403382188 ],
			"zoom" : 0.5199999999999998
		}

    # 应用视角设置
    ctr = vis.get_view_control()
    ctr.set_lookat(parameters["lookat"])
    ctr.set_front(parameters["front"])
    ctr.set_up(parameters["up"])
    ctr.set_zoom(parameters["zoom"])

    # 渲染点云为图像
    vis.poll_events()
    vis.update_renderer()

    # 获取点云的2D图像
    depth_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    # 调整尺寸并格式化图像
    depth_image = (depth_image * 255).astype(np.uint8)  # 转换为8位图像
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以便与OpenCV兼容

    return depth_image

def depth_to_point_cloud(depth, K_cam):
    if isinstance(K_cam, np.ndarray):
        K_cam = torch.from_numpy(K_cam).to(depth.device)

    A, B, H, W = depth.shape
    i, j = torch.meshgrid(
        torch.arange(0, H, device=depth.device),
        torch.arange(0, W, device=depth.device),
        indexing='ij'
    )
    z = depth[0][0]
    x = (j - K_cam[0, 2]) * z / K_cam[0, 0]
    y = (i - K_cam[1, 2]) * z / K_cam[1, 1]
    # y = -y

    points = torch.stack((x, y, z), dim=2).reshape(-1, 3)
    return points.detach().cpu().numpy()

def depth_to_colored_point_cloud(depth, K_cam, bgr_image):
    """
    Converts a depth map to a 3D point cloud and assigns RGB colors.

    Args:
        depth (torch.Tensor): Shape (1, 1, H, W), depth values.
        K_cam (np.ndarray or torch.Tensor): 3x3 intrinsic matrix.
        bgr_image (np.ndarray): (H, W, 3) image in BGR format (from OpenCV).

    Returns:
        points (np.ndarray): (N, 3) 3D points.
        colors (np.ndarray): (N, 3) RGB colors in [0, 1].
    """
    if isinstance(K_cam, np.ndarray):
        K_cam = torch.from_numpy(K_cam).to(depth.device)

    _, _, H, W = depth.shape

    # Pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(0, H, device=depth.device),
        torch.arange(0, W, device=depth.device),
        indexing='ij'
    )

    z = depth[0, 0]
    x = (j - K_cam[0, 2]) * z / K_cam[0, 0]
    y = (i - K_cam[1, 2]) * z / K_cam[1, 1]
    y = -y  # flip if needed
    x = -x  # flip if needed

    points = torch.stack((x, y, z), dim=2).reshape(-1, 3).detach().cpu().numpy()

    # Convert BGR to RGB and normalize
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colors = rgb_image.reshape(-1, 3)

    return points, colors


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pull_K_cam(calib_path):
    filedata = read_calib_file(calib_path)
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    K_cam = P_rect_20[0:3, 0:3]
    return K_cam


def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def in_h_range_points(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180),
                          np.arctan2(n, m) < (-fov[0] * np.pi / 180))


def in_v_range_points(points, m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180),
                          np.arctan2(n, m) > (fov[0] * np.pi / 180))


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """

    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points

    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]


def in_range_points(points, size):
    """ extract in-range points """
    return np.logical_and(points > 0, points < size)


def velo_points_filter(points, v_fov, h_fov):
    """ extract points corresponding to FOV setting """

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)

    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:, None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:, None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:, None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat), axis=0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    color = depth_color(dist_lim, 0, 70)

    return xyz_, color


def calib_velo2cam(filepath):
    """ 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


def calib_cam2cam(filepath, mode):
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_


def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02', image_shape=None):
    """ 
    Return velodyne 3D points corresponding to camera 2D image and sparse depth map
    """

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(vc_path)

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(cc_path, mode)
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)

    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # Initialize sparse depth map
    if image_shape is None:
        raise ValueError(
            "Image shape must be provided to generate sparse depth map")

    # Create a depth map with NaN values
    depth_map = np.full(image_shape[:2], 0)

    # Convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    xyz_c = np.delete(xyz_v, 3, axis=0)

    # Convert camera coordinates(X_c, Y_c, Z_c) to image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    # Normalize by the third coordinate to get 2D pixel coordinates
    xy_i = xyz_c[:2, :] / xyz_c[2, :]
    depth_values = xyz_c[2, :]  # Z-coordinate (depth) in camera space

    # Filter out points that are out of image bounds
    valid_mask = np.logical_and.reduce((
        xy_i[0, :] >= 0,
        xy_i[0, :] < image_shape[1],  # x coordinate within image width
        xy_i[1, :] >= 0,
        xy_i[1, :] < image_shape[0],  # y coordinate within image height
    ))

    valid_points = xy_i[:, valid_mask]
    valid_depths = depth_values[valid_mask]

    # Fill the depth map with depth values
    for i in range(valid_points.shape[1]):
        x = int(valid_points[0, i])
        y = int(valid_points[1, i])
        depth_map[y, x] = valid_depths[i]

    # Return both the projected points and the sparse depth map
    return valid_points, c_, depth_map


def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(
            points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(
            points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :3]


def set_axes_equal(ax):
    """使 3D 图的刻度长短一致"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # 找到所有坐标的中心和范围
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # 计算出最大的范围
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    # 设置每个坐标轴的范围，使其相等
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def save_point_cloud_to_ply_open3d(point_cloud, image, filename):
    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点云的坐标
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 将图像展开为 (N, 3) 形式的颜色数据
    colors = image.reshape(-1, 3) / 255.0  # 归一化颜色到 [0, 1]

    # 设置点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存点云为 .ply 文件
    o3d.io.write_point_cloud(filename, pcd)
    return pcd

def visualize_pointcloud(points: np.ndarray, colors: np.ndarray = None):
    """
    Visualizes a 3D point cloud using Open3D.

    Args:
        points (np.ndarray): Nx3 array of XYZ coordinates.
        colors (np.ndarray, optional): Nx3 array of RGB colors in [0, 1]. Defaults to None.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points should be an Nx3 NumPy array.")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError("Colors must have the same shape as points.")
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a coordinate frame at the origin (0, 0, 0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    def set_top_down_view(vis):
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])   # Camera looks along -Z
        ctr.set_up([0.0, 1.0, 0.0])       # Y axis is 'up' in the view
        ctr.set_lookat(pcd.get_center())  # Focus on the center of the cloud
        ctr.set_zoom(0.5)                 # Zoom level, adjust as needed
        return False  # Return False to stop calling repeatedly

    # Wrap inside a key callback so it remains interactive
    key_to_callback = {ord("T"): set_top_down_view}

    print("Press 'T' in the viewer to set top-down view.")
    o3d.visualization.draw_geometries_with_key_callbacks([pcd, axis], key_to_callback)

    # o3d.visualization.draw_geometries([pcd])

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        net = run.net_ema.module.cuda()
        net.eval()
        # 读取左目图像
        global_path = "/home/kitti_dataset/"
        base = global_path + "2011_09_26/2011_09_26_drive_0005_sync"

        image_type = 'color'  # 'grayscale' or 'color' image

        mode = '00' if image_type == 'grayscale' else '02'

        v2c_filepath = global_path + '2011_09_26/calib_velo_to_cam.txt'
        c2c_filepath = global_path + '2011_09_26/calib_cam_to_cam.txt'
        image_mean = np.array([90.9950, 96.2278, 94.3213])
        image_std = np.array([79.2382, 80.5267, 82.1483])
        image_height = 352
        image_width = 1216

        t_file = base + "/image_" + mode + "/timestamps.txt"
        with open(t_file, 'r') as f:
            N = sum(1 for _ in f)

        viewer = ImageGridViewer(titles=["Depth", "Raw", "Lidar", "Projected"])
        # viewer_3d = PointCloudImageViewer()
        # viewer_pc = PointCloudViewer()

        for i in tqdm(range(N)):
            
            image = np.array(Image.open(os.path.join(
                base, 'image_' + mode + f'/data/0000000{i:03d}.png')).convert('RGB'), dtype=np.uint8)
            
            if image is None:
                break
            width, height = image.shape[1], image.shape[0]

            # bin file -> numpy array
            velo_points = load_from_bin(os.path.join(
                base, f'velodyne_points/data/0000000{i:03d}.bin'))

            image_type = 'color'  # 'grayscale' or 'color' image
            # image_00 = 'grayscale image' , image_02 = 'color image'
            mode = '00' if image_type == 'grayscale' else '02'

            ans, c_, lidar = velo3d_2_camera2d_points(velo_points, v_fov=(-24.9, 2.0), h_fov=(-45, 45),
                                                      vc_path=v2c_filepath, cc_path=c2c_filepath, mode=mode,
                                                      image_shape=image.shape)

            image_vis = print_projection_plt(points=ans, color=c_, image=image.copy())

            # depth completion
            K_cam = torch.from_numpy(pull_K_cam(
                c2c_filepath).astype(np.float32)).cuda()
            
            tp = image.shape[0] - image_height
            lp = (image.shape[1] - image_width) // 2
            image = image[tp:tp + image_height, lp:lp + image_width]
            lidar = lidar[tp:tp + image_height, lp:lp + image_width, None]
            image_vis = image_vis[tp:tp + image_height, lp:lp + image_width]
            K_cam[0, 2] -= lp
            K_cam[1, 2] -= tp

            image = (image - image_mean) / image_std

            image_tensor = image.transpose(2, 0, 1).astype(np.float32)[None]
            lidar_tensor = lidar.transpose(2, 0, 1).astype(np.float32)[None]

            image_tensor = torch.from_numpy(image_tensor)
            lidar_tensor = torch.from_numpy(lidar_tensor)

            output = net(image_tensor.cuda(), None,
                         lidar_tensor.cuda(), K_cam[None].cuda())

            if isinstance(output, (list, tuple)):
                output = output[-1]

            tensor_output = output
            
            output = output.squeeze().detach().cpu().numpy()
            image = image * image_std + image_mean
            
            output_max, output_min = output.max(), output.min()
            output_norm = (output - output_min) / (output_max - output_min) * 255
            output_norm = output_norm.astype('uint8')
            output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_JET)

            # Transform from depth to 3D
            # points_3d = depth_to_point_cloud(tensor_output, K_cam)
            points_3d, colors = depth_to_colored_point_cloud(tensor_output, K_cam, image.astype(np.uint8)[:, :, ::-1])
            del tensor_output # free GPU memory

            visualize_pointcloud(points_3d, colors)

            # viewer_3d.visualize_for_seconds(points_3d, output_color, 2)

            # viewer_pc.visualize_for_seconds(points_3d, 2, colors)

            viewer.update([output_color, image.astype(np.uint8)[:, :, ::-1], lidar.astype(np.uint8) * 3, image_vis],
                           duration=0.5)  # Display for x seconds

            cv2.imwrite(f'outputs/0000000{i:03d}_depth.png', output_color)
            
            cv2.imwrite(f'outputs/0000000{i:03d}_image.png', image.astype(np.uint8)[:, :, ::-1])
            cv2.imwrite(f'outputs/0000000{i:03d}_lidar.png', lidar.astype(np.uint8) * 3)
            cv2.imwrite(f'outputs/0000000{i:03d}_image_vis.png', image_vis)


if __name__ == '__main__':
    main()
