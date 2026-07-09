import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.cm as cm
import time
import threading

class ImageGridViewer:
    def __init__(self, titles=None):
        """
        Initializes a persistent 2x2 image grid viewer that preserves image size.
        
        Args:
            titles (list of str): Optional list of 4 titles for the subplots.
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.titles = titles if titles else [""] * 4

        self.images = []
        for ax, title in zip(self.axes.flatten(), self.titles):
            ax.set_title(title)
            ax.axis('off')
            ax.set_aspect('equal')  # Prevent image distortion

            # Initialize with dummy image to match interface
            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            im = ax.imshow(dummy_img, interpolation='none')
            self.images.append(im)

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.ion()
        plt.show()

    def update(self, images, duration=0):
        """
        Updates the grid with new images and refreshes the display.
        
        Args:
            images (list of np.ndarray): List of 4 OpenCV (BGR) images.
            duration (float): Seconds to pause after displaying (0 = no pause).
        """
        if len(images) != 4:
            raise ValueError("Exactly 4 images required.")

        for i in range(4):
            rgb_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            self.images[i].set_data(rgb_img)
            self.images[i].set_extent((0, rgb_img.shape[1], rgb_img.shape[0], 0))  # Preserve size

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if duration > 0:
            plt.pause(duration)

class PointCloudImageViewer:
    def __init__(self, window_title="Viewer", width=1280, height=720):
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(window_title, width, height)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        self.panel = gui.Horiz()
        self.window.add_child(self.panel)

        # Placeholder image
        dummy_img = o3d.geometry.Image(np.zeros((480, 640, 3), dtype=np.uint8))
        self.image_widget = gui.ImageWidget(dummy_img)

        self.panel.add_child(self.image_widget)
        self.panel.add_child(self.scene_widget)

        self.geometry_name = "pointcloud"

    def update(self, pointcloud_np, image_cv2):
        # Convert image
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        o3d_image = o3d.geometry.Image(image_rgb)

        # Update image widget
        self.image_widget.update_image(o3d_image)

        # Create new point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_np)

        # Clear old and add new geometry
        self.scene_widget.scene.clear_geometry()
        self.scene_widget.scene.add_geometry(self.geometry_name, pcd, rendering.MaterialRecord())

        # Recenter camera
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def run(self):
        gui.Application.instance.run()

    def run_one_tick(self):
        self.app.run_one_tick()

    def visualize_for_seconds(self, pointcloud_np, image_cv2, duration_sec=5):
        self.update(pointcloud_np, image_cv2)
        start = time.time()
        while time.time() - start < duration_sec:
            self.app.run_one_tick()
            time.sleep(0.01)  # to prevent max CPU use

    def run_with_updates(self, data_loader_func, interval_sec=2):
        def update_loop():
            while True:
                pcd_np, img_cv2 = data_loader_func()
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.update(pcd_np, img_cv2)
                )
                time.sleep(interval_sec)

        threading.Thread(target=update_loop, daemon=True).start()
        self.app.run()

class PointCloudViewer:
    def __init__(self, window_title="PointCloud Viewer", width=800, height=600):
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(window_title, width, height)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        self.geometry_name = "pointcloud"

    def _color_by_x(self, points_np):
        x = points_np[:, 0]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)  # Normalize x to [0,1]
        colors = cm.viridis(x_norm)[:, :3]  # Drop alpha channel
        return colors

    def update(self, points_np, colors = None):
        # Create and set point cloud geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # Color by X values
        if colors is None:
            colors_np = self._color_by_x(points_np)
        else:
            colors_np = colors
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

        self.scene_widget.scene.clear_geometry()
        self.scene_widget.scene.add_geometry(self.geometry_name, pcd, rendering.MaterialRecord())

        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        diameter = np.linalg.norm(extent)

        lookat = pcd.get_center()
        front = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
        zoom_distance = 0.2 * np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())

        eye = lookat - front * zoom_distance
        self.scene_widget.scene.camera.look_at(lookat, eye, up)

    def visualize_for_seconds(self, pointcloud_np, duration_sec=2, colors_np=None):
        self.update(pointcloud_np, colors_np)
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            self.app.run_one_tick()
            time.sleep(0.01)

    def run(self):
        self.app.run()
