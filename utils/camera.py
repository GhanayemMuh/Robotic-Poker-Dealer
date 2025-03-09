import os

import pyrealsense2 as rs
import numpy as np
import json
import cv2


class Camera:
    COMMUNITY_ROI = (270, 690, 1800, 1080)  # (min_x, min_y, max_x, max_y)

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable RGB and Depth streams
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.pipeline.stop()

        try:
            # Start the pipeline
            self.pipeline.start(self.config)
            print("Camera pipeline started successfully.")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to start the RealSense pipeline: {e}")

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Initialize intrinsics and distortion coefficients
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = None

        # Attempt to load intrinsics and custom configuration at initialization
        self.load_intrinsics()
        self.load_custom_configuration("v1.json")

        self.frame = None
        self.debug_frame = None
        self.depth_frame = None
        self.baseline_depth = None  # Initialize baseline depth for ROI comparison

    def load_custom_configuration(self, config_file):
        """
        Load a custom RealSense camera configuration from a JSON file.

        Args:
            config_file (str): Path to the JSON file containing the configuration.

        Returns:
            None
        """
        if not os.path.exists(config_file):
            print(f"Configuration file {config_file} not found. Skipping custom configuration.")
            return

        try:
            with open(config_file, "r") as file:
                config_data = json.load(file)

            device = self.profile.get_device()
            advanced_mode = rs.rs400_advanced_mode(device)

            if not advanced_mode.is_enabled():
                advanced_mode.toggle_advanced_mode(True)
                print("Enabled advanced mode on the RealSense device.")

            advanced_mode.load_json(json.dumps(config_data))
            print(f"Custom configuration from {config_file} loaded successfully.")

        except Exception as e:
            print(f"Failed to load custom configuration: {e}")

    def draw_regions_on_frame(self, frame):
        """
        Draw the community ROI and player split lines on the frame.

        Args:
            frame (np.ndarray): The RGB frame to annotate.

        Returns:
            np.ndarray: Annotated frame.
        """
        # Draw the community ROI
        x_min, y_min, x_max, y_max = Camera.COMMUNITY_ROI
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box for community ROI

        # Draw the line to split the frame into player regions
        frame_height, frame_width, _ = frame.shape
        mid_x = frame_width // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (255, 0, 0), 2)  # Blue vertical line

        return frame

    def monitor_depth_region(self, roi_rgb, threshold=50):
        """
        Monitor a region in the depth frame for changes compared to the baseline.

        Args:
            roi_rgb (tuple): (x_min, y_min, x_max, y_max) in the RGB frame.
            threshold (int): Depth variance threshold for detecting changes.

        Returns:
            bool: True if significant change is detected, False otherwise.
        """
        # Project the RGB ROI to Depth ROI
        roi_depth = self.project_rgb_to_depth(roi_rgb)

        # Get the current depth frame
        depth_frame = self.get_depth_frame()
        if depth_frame is None or self.baseline_depth is None:
            raise RuntimeError("Baseline depth not initialized or invalid depth frame")

        # Extract the relevant ROI regions
        x_min, y_min, x_max, y_max = roi_depth
        current_roi = depth_frame[y_min:y_max, x_min:x_max]
        baseline_roi = self.baseline_depth

        # Compute the absolute difference
        depth_diff = np.abs(current_roi.astype(np.float32) - baseline_roi.astype(np.float32))
        return np.max(depth_diff) > threshold

    def capture_baseline_depth(self, roi_rgb=None):
        """
        Capture and store the baseline depth frame for a region defined in the RGB frame.

        Args:
            roi_rgb (tuple): (x_min, y_min, x_max, y_max) in the RGB frame. If None, uses COMMUNITY_ROI.

        Returns:
            None
        """
        if roi_rgb is None:
            roi_rgb = Camera.COMMUNITY_ROI

        # Project the RGB ROI to the Depth frame
        roi_depth = self.project_rgb_to_depth(roi_rgb)

        # Get the current depth frame
        depth_frame = self.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Failed to capture baseline depth frame")

        # Extract and store the depth data for the projected ROI
        x_min, y_min, x_max, y_max = roi_depth
        self.baseline_depth = depth_frame[y_min:y_max, x_min:x_max].copy()
        print(f"Baseline depth frame captured for ROI: {roi_depth}")

    def project_rgb_to_depth(self, roi_rgb):
        """
        Project an ROI from the RGB frame to the Depth frame.

        Args:
            roi_rgb (tuple): (x_min, y_min, x_max, y_max) in RGB frame.

        Returns:
            tuple: (x_min, y_min, x_max, y_max) in Depth frame.
        """
        rgb_width, rgb_height = 1920, 1080  # RGB frame resolution
        depth_width, depth_height = 1280, 720  # Depth frame resolution

        # Calculate scaling factors
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height

        # Project ROI to depth frame
        x_min, y_min, x_max, y_max = roi_rgb
        roi_depth = (
            int(x_min * scale_x),
            int(y_min * scale_y),
            int(x_max * scale_x),
            int(y_max * scale_y),
        )
        return roi_depth

    def load_intrinsics(self, calibration_file="hand_eye_calibration_results.npz"):
        """
        Load camera intrinsics and distortion coefficients from a file.
        If the file does not exist, the values remain None.
        """
        if os.path.exists(calibration_file):
            print("Calibration file found. Loading intrinsics...")
            data = np.load(calibration_file)
            self.color_intrinsics = data["camera_matrix"]
            print("Intrinsics loaded successfully.")
        else:
            print("Calibration file not found. Intrinsics remain None.")

    def get_rgb_frame(self):
        """Get a single RGB frame."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Could not retrieve RGB frame")

        # Convert the frame to a NumPy array
        self.frame = np.asanyarray(color_frame.get_data())
        self.debug_frame = self.frame.copy()

        # Annotate the frame with regions
        self.debug_frame = self.draw_regions_on_frame(self.debug_frame)

        return self.frame

    def get_depth_frame(self):
        """Get a single depth frame."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            raise RuntimeError("Could not retrieve depth frame")

        self.depth_frame = depth_frame

        return np.asanyarray(depth_frame.get_data())

    def pixel_to_3d(self, pixel, depth_frame):
        """Given a pixel (x, y) and depth frame, return the corresponding 3D point in the camera's coordinate system.

        Args:
            pixel (tuple): A tuple (x, y) representing the pixel coordinates.
            depth_frame (pyrealsense2.depth_frame): The depth frame.

        Returns:
            tuple: A tuple (X, Y, Z) representing the 3D point.
        """
        x, y = pixel

        # Get depth value at the specified pixel
        depth_value = depth_frame.get_distance(x, y)
        if depth_value <= 0:
            raise ValueError("Depth value is zero or invalid at the specified pixel")

        # Convert pixel to 3D point
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth_value)
        return tuple(depth_point)

    def stop(self):
        """Stop the camera pipeline."""
        try:
            self.pipeline.stop()
            print("Camera pipeline stopped.")
        except RuntimeError as e:
            print(f"Failed to stop the pipeline: {e}")
