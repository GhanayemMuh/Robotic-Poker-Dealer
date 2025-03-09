import time

import cv2
import numpy as np
import joblib
import pyrealsense2 as rs

class PokerChips:
    def __init__(self, model_path="chip_color_model.pkl"):
        """
        Initialize the PokerChips class.

        Args:
            model_path (str): Path to the trained model for classification.
        """
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

    def detect_and_classify_stacks(self, camera, arm, region, total_frames=20, min_detections=5, debug=False):
        """
        Detect and classify poker chip stacks over a series of frames in the specified region.

        Args:
            camera (Camera): Instance of the Camera class.
            arm (Arm): Instance of the Arm class to position the robot.
            region (str): Region to monitor ("pot", "player1", or "player2").
            total_frames (int): Number of consecutive frames to process.
            min_detections (int): Minimum number of frames a stack must appear in to be confirmed.
            debug (bool): If True, display the frames with detected stacks.

        Returns:
            list: A list of confirmed stacks, each containing its center (x, y, z) in the base frame and color.
        """

        # 1) Move the arm to the specified region
        arm.move_to_stacks(region)
        time.sleep(1)

        # 2) Get the *current* arm pose in radians (ensures correct yaw/orientation).
        code, current_pose = arm.arm.get_position(is_radian=True)
        if code != 0:
            print("Failed to get end-effector position after moving to region.")
            return []

        # 3) Convert that pose into a T_base2ee matrix.
        #    IMPORTANT: Ensure this matrix is *actually* EE→Base if you multiply by an EE point later.
        #    If your function returns Base→EE, you would need its inverse.
        T_base2ee_current = arm.euler_to_transformation_matrix(current_pose)

        # Prepare to store detections from all frames
        detections_across_frames = []

        # 4) Collect multiple frames for robust detection
        for frame_idx in range(total_frames):
            # Grab fresh frames
            rgb_frame = camera.get_rgb_frame()
            depth_frame = camera.get_depth_frame()

            if rgb_frame is None or depth_frame is None:
                print("Failed to retrieve frame(s) from the camera.")
                continue

            # Convert to grayscale for Hough circle detection
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

            # 5) Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT_ALT,
                dp=1.1,
                minDist=20,
                param1=200,
                param2=0.8,
                minRadius=65,
                maxRadius=130
            )

            frame_detections = []
            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))

                for (x, y, r) in circles:
                    # Define an ROI around the circle
                    x1, y1 = max(0, x - r), max(0, y - r)
                    x2, y2 = min(x + r, rgb_frame.shape[1]), min(y + r, rgb_frame.shape[0])
                    roi = rgb_frame[y1:y2, x1:x2]

                    if roi.size == 0:
                        continue

                    # Create a circular mask for the ROI
                    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (r, r), r, 255, -1)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

                    # Compute HSV histogram features
                    hsv = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    feature_vector = hist.flatten().reshape(1, -1)

                    # Predict chip color
                    predicted_label = self.model.predict(feature_vector)[0]

                    # Extract depth information
                    depth_array = np.asanyarray(depth_frame)
                    h_d, w_d = depth_array.shape
                    x_min, x_max = max(0, x - 3), min(w_d, x + 3)
                    y_min, y_max = max(0, y - 3), min(h_d, y + 3)
                    depth_region = depth_array[y_min:y_max, x_min:x_max]

                    # Calculate the average depth value, ignoring zeros
                    valid_depths = depth_region[depth_region > 0]
                    if valid_depths.size == 0:
                        print(f"No valid depth values around pixel ({x}, {y}).")
                        continue
                    depth = np.mean(valid_depths) * camera.depth_scale

                    # Deprojection: Pixel (x, y) + depth -> 3D in Camera frame
                    depth_intrinsics = camera.depth_frame.profile.as_video_stream_profile().get_intrinsics()
                    point_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)

                    # Convert to homogeneous coords for 4×4 transforms
                    point_camera_hom = np.array([
                        point_camera[0],
                        point_camera[1],
                        point_camera[2],
                        1.0
                    ]).reshape(4, 1)

                    base_to_cam = np.dot(T_base2ee_current,arm.T_cam2ee)
                    # 6) Transform from Camera Frame → EE Frame (assuming arm.T_cam2ee is 4×4 camera->EE)

                    #point_ee_hom = np.dot(arm.T_cam2ee, point_camera_hom)

                    # 7) Transform from EE Frame → Base Frame using the full 4×4 matrix
                    #    Provided T_base2ee_current is actually EE->Base
                    point_base_hom = np.dot(base_to_cam, point_camera_hom)

                    # Extract (x, y, z) in base frame
                    point_base = point_base_hom[:3].flatten()

                    # Store detection
                    frame_detections.append({
                        'center': tuple(point_base),
                        'color': predicted_label
                    })

                    # Debug visualization
                    if debug:
                        cv2.circle(rgb_frame, (x, y), r, (0, 255, 0), 2)
                        cv2.putText(rgb_frame, predicted_label,
                                    (x - r, y - r - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

            # If debug, display the current frame
            if debug:
                cv2.imshow("Detected Stacks", rgb_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            detections_across_frames.append(frame_detections)
            time.sleep(0.2)  # Control loop speed as needed

        # ----- Clustering detections across frames -----
        clusters = []
        distance_thresh = 10 / 1000  # 10 mm threshold in base frame

        for frame_detections in detections_across_frames:
            for detection in frame_detections:
                x_base, y_base, _ = detection['center']
                assigned = False

                # Try to add detection into an existing cluster if close enough
                for cluster in clusters:
                    cluster_x = np.mean([d['center'][0] for d in cluster])
                    cluster_y = np.mean([d['center'][1] for d in cluster])
                    dist = np.sqrt((x_base - cluster_x) ** 2 + (y_base - cluster_y) ** 2)
                    if dist <= distance_thresh:
                        cluster.append(detection)
                        assigned = True
                        break

                # Create a new cluster if no match found
                if not assigned:
                    clusters.append([detection])

        # ----- Confirm clusters with at least min_detections in total_frames -----
        confirmed_stacks = []
        for cluster in clusters:
            if len(cluster) >= min_detections:
                # Average cluster coordinates
                avg_x = np.mean([d['center'][0] for d in cluster])
                avg_y = np.mean([d['center'][1] for d in cluster])
                avg_z = np.mean([d['center'][2] for d in cluster])

                # Determine majority color in that cluster
                colors = [d['color'] for d in cluster]
                majority_color = max(set(colors), key=colors.count)

                confirmed_stacks.append({
                    'center': (avg_x, avg_y, avg_z),
                    'color': majority_color
                })

        if debug:
            cv2.destroyAllWindows()

        print(f"Detected and confirmed {len(confirmed_stacks)} stacks in region '{region}'.")
        return confirmed_stacks

