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

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import struct # to decode pointcloud_2 data

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
    y = -y

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
    # dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    p3N = xyz_[:3, :]
    norms = np.linalg.norm(p3N, axis=0)
    color = depth_color(norms, 0, np.max(norms))

    return xyz_, color


def calib_velo2cam():
    """ 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """

    R = np.array([-0.1533276, -0.98776117, 0.02860976, 
                0.04630053, -0.0361014, -0.99827499, 
                0.98709012, -0.15173846, 0.05126921]
                )
    R = R.reshape(3, 3)
    T = np.array([0.28564802, 1.01355072, -0.9722258])
    T = T.reshape(3, 1)
    return R, T

def calib_cam2cam():
    P_ = np.array([659.29813, 0.0, 760.64067,
                0.0, 662.57773, 543.84931, 
                0.0, 0.0, 1.0]
                )
    P_ = P_.reshape(3, 3)
    d = np.array([-0.193406, 0.026607, 0.001749, -0.000172, 0.0])

    return P_, d


def velo3d_2_camera2d_points(points, v_fov, h_fov, K, D=None, image_shape=None):
    """ 
    Project 3D Velodyne points to 2D image points using camera calibration with distortion correction.
    """

    # Get calibration: rotation & translation from Velodyne to Camera
    R_vc, T_vc = calib_velo2cam()  # R: (3x3), T: (3x1)

    # Filter points based on FOV
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)  # xyz_v: 4xN (homogeneous)

    if image_shape is None:
        raise ValueError("Image shape must be provided to generate sparse depth map")

    # Convert xyz_v (homogeneous) to 3xN
    xyz_v = xyz_v[:3, :]

    # Transform points to camera coordinates: X_c = R * X_v + T
    xyz_c = R_vc @ xyz_v + T_vc

    # Transpose to Nx3 for OpenCV
    xyz_c = xyz_c.T.reshape(-1, 1, 3)

    # Project to 2D with distortion
    if D is not None:
        img_points, _ = cv2.projectPoints(xyz_c, np.zeros((3,1)), np.zeros((3,1)), K, D)
    else:
        img_points, _ = cv2.projectPoints(xyz_c, np.zeros((3,1)), np.zeros((3,1)), K, None)
    img_points = img_points.reshape(-1, 2)

    # Extract depth (Z in camera frame)
    depth_values = xyz_c[:, 0, 2]

    # Create empty depth map
    depth_map = np.full(image_shape[:2], 0, dtype=np.float32)

    # Filter valid points within image bounds
    valid_mask = np.logical_and.reduce((
        img_points[:, 0] >= 0,
        img_points[:, 0] < image_shape[1],
        img_points[:, 1] >= 0,
        img_points[:, 1] < image_shape[0],
    ))

    valid_img_points = img_points[valid_mask]
    valid_depths = depth_values[valid_mask]

    # Populate depth map
    for pt, depth in zip(valid_img_points, valid_depths):
        x, y = int(pt[0]), int(pt[1])
        depth_map[y, x] = depth

    return valid_img_points.T, c_, depth_map

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

def pointcloud2_to_xyz_array(cloud_msg):
    """
    Convert sensor_msgs/PointCloud2 to an Nx3 NumPy array.
    Assumes XYZ floats.
    """
    fmt = 'fff'  # just XYZ (each float32)
    width = cloud_msg.width
    height = cloud_msg.height
    point_step = cloud_msg.point_step
    row_step = cloud_msg.row_step
    data = cloud_msg.data

    points = []
    for i in range(0, len(data), point_step):
        x, y, z = struct.unpack_from(fmt, data, offset=i)
        points.append((x, y, z))

    return np.array(points, dtype=np.float32)

def load_image_from_rosbag(bag_path, topic, stamps, N=-1):
    """
    Loads each rgb image from a rosbag file taking also into account a simple synchronization
    strategy w.r.t to the pcd timestamps.

    Args:
        bag_path (str): path to the rosbag file.
        topic (str): the topic with image data (sensor_msgs/ImageRaw) 
        stamps (list(float)): time stamp of each pointcloud/LiDAR frame.
        N (int): optional max number of images retrieved

    Returns:
        images (list(np.ndarray)): list of BGR images.
        pcd_ids (list(int)): index of the closest (in time) pointclouds for the chosen images.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    images = []
    pcd_ids = []

    with Reader(bag_path) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        c = 0   
        stamp_i = 0
        start_found = False

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            
            now = timestamp*1e-9

            if now > stamps[min(stamp_i,len(stamps)-1)] and not start_found:
                stamp_i += 1
                continue
            else:
                start_found = True

            print(f"now: {now}")
            print(f"stamp: {stamps[stamp_i]}")

            if now >= stamps[stamp_i] and now < stamps[min(stamp_i+1,len(stamps)-1)]:

                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                if msg.encoding != "bayer_rggb8":
                    print(f"Unsupported encoding: {msg.encoding}")
                    continue

                height = msg.height
                width = msg.width
                data = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))

                # Debayer using OpenCV: Bayer RGGB -> RGB
                rgb_img = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)
                images.append(rgb_img)
                pcd_ids.append(stamp_i)
                stamp_i += 1

                c += 1
                if c > N and N > 0:
                    break

    if len(images) > 0:
        print(f"Loaded images shape: {images[0].shape}, dtype: {images[0].dtype}")
    return images, pcd_ids


def load_pcd_from_rosbag(bag_path, topic, N=-1):
    """
    Loads each pcd from a rosbag file.

    Args:
        bag_path (str): path to the rosbag file.
        topic (str): the topic with pointcloud data (sensor_msgs/PointCloud2) 
        N (int): optional max number of pcd retrieved

    Returns:
        pcds (list(np.ndarray)): list of pointclouds (pcd).
        stamps (list(float)): time stamp of each pointcloud/LiDAR frame.
    """
    # Create a typestore for the matching ROS release.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Topic to filter
    pcds = []
    stamps = []

    c = 0
    # Create reader instance and open for reading.
    with Reader(bag_path) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        for connection, timestamp, rawdata in reader.messages(connections=connections):

            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            np_points = pointcloud2_to_xyz_array(msg)
            pcds.append(np_points) 
            stamps.append(timestamp*1e-9)
            c += 1
            if c > N and N > 0:
                break
    
    return pcds, stamps


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

        global_path = "/home/kitti_dataset/2024_11_07_born/"
        image_bag_name = "2024-11-07-11-24-29_3.bag"
        pcd_bag_name = "2024-11-07-11-24-14_1.bag"

        topic_lidar = "/ona2/sensors/pandar_front/cloud"
        topic_camera = "/ona2/sensors/flir_camera_front/image_raw"

        image_mean = np.array([90.9950, 96.2278, 94.3213])
        image_std = np.array([79.2382, 80.5267, 82.1483])
        image_height = 352
        image_width = 1216

        # Get LiDAR data from rosbag
        pcd_array, stamps = load_pcd_from_rosbag(global_path + "robot/" + pcd_bag_name, topic_lidar)

        # Get image data from rosbag
        images_array, pcd_ids = load_image_from_rosbag(global_path + "camera/" + image_bag_name, topic_camera, stamps)

        print(f"Loaded {len(images_array)} images and {len(pcd_array)} pcds.")

        viewer = ImageGridViewer(titles=["Depth", "Raw", "Lidar", "Projected"])
        # viewer_3d = PointCloudImageViewer()
        # viewer_pc = PointCloudViewer()

        N = len(images_array)
        for i in tqdm(range(N)): 

            image = images_array[i]
            w, h = image.shape[1], image.shape[0]

            # Undistort image
            K, D = calib_cam2cam()

            # h, w = img.shape[:2]
            # new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

            # # Compute undistort maps
            # map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

            # # Apply remap
            # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                # Get optimal new camera matrix
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

                # Undistort using new_K
            undistorted_img = cv2.undistort(image, K, D, None, new_K)
            x, y, w, h = roi
            cropped_img = undistorted_img[y:y+h, x:x+w]

            # Resize image to model dimensions
            # resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            resized = cv2.resize(undistorted_img, (image_width, image_height), interpolation=cv2.INTER_AREA)
            image = resized

            # Correct intrinsics
            CALIB_W = 1440
            CALIB_H = 1080

            sx = image_width / CALIB_W
            sy = image_height / CALIB_H

            new_K[0,0] *= sx 
            new_K[1,1] *= sy 
            new_K[0,2] *= sx 
            new_K[1,2] *= sy 

            # K[0,0] *= sx 
            # K[1,1] *= sy 
            # K[0,2] *= sx 
            # K[1,2] *= sy 

            # Get pointcloud
            pcd = pcd_array[pcd_ids[i]]

            # Project 3D points into 2D (in pixels)
            # ans, c_, lidar = velo3d_2_camera2d_points(pcd, v_fov=(-24.9, 2.0), h_fov=(-45, 45),
            #                                           K=K, D=D, image_shape=image.shape)
            ans, c_, lidar = velo3d_2_camera2d_points(pcd, v_fov=(-24.9, 2.0), h_fov=(-45, 45),
                                                      K=new_K, D=None, image_shape=image.shape)

            image_vis = print_projection_plt(points=ans, color=c_, image=image.copy())

            # Depth completion
            K_cam = torch.from_numpy(new_K.astype(np.float32)).cuda()
            # K_cam = torch.from_numpy(K.astype(np.float32)).cuda()
            
            lidar = lidar[:, :, None]

            image = (image - image_mean) / image_std

            image_tensor = image.transpose(2, 0, 1).astype(np.float32)[None]
            lidar_tensor = lidar.transpose(2, 0, 1).astype(np.float32)[None]

            image_tensor = torch.from_numpy(image_tensor)
            lidar_tensor = torch.from_numpy(lidar_tensor)

            m_start = time.time()

            output = net(image_tensor.cuda(), None,
                         lidar_tensor.cuda(), K_cam[None].cuda())

            m_end = time.time()
            print(f"Inference time: {(m_end - m_start)*1000.0} ms")

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

            # viewer_pc.visualize_for_seconds(points_3d, 5, colors)

            lidar = depth_color(lidar, 0, 35)
            lidar = cv2.applyColorMap(lidar.astype(np.uint8)*3, cv2.COLORMAP_TURBO)
            viewer.update([output_color, image.astype(np.uint8)[:, :, ::-1], lidar, image_vis],
                           duration=0.1)  # Display for x seconds

            # (Optional) Save outputs
            # cv2.imwrite(f'outputs/0000000{i:03d}_depth.png', output_color)
            
            # cv2.imwrite(f'outputs/0000000{i:03d}_image.png', image.astype(np.uint8)[:, :, ::-1])
            # cv2.imwrite(f'outputs/0000000{i:03d}_lidar.png', lidar.astype(np.uint8) * 3)
            # cv2.imwrite(f'outputs/0000000{i:03d}_image_vis.png', image_vis)

if __name__ == '__main__':
    main()
