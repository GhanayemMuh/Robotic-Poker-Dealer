import argparse
import os
import cv2
import numpy as np
from glob import glob
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # for saving/loading the model

import pyrealsense2 as rs




def extract_chip_roi_and_features(image_bgr, show_detection=False):
    """
    Detect the poker chip (circle) in the given BGR image,
    and compute a color-histogram-based feature vector from the chip region.

    If `show_detection` is True, the detected circle will be drawn on the image
    and displayed for 500 ms.

    Returns:
        feature_vector (np.array of shape [hist_bins_total]): The chip's color histogram.
        None if no circle was detected or feature extraction failed.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 5, 75, 75)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.1,
        minDist=20,
        param1=200,
        param2=0.8,
        minRadius=50,
        maxRadius=300
    )

    if circles is None:
        if show_detection:
            cv2.imshow("Detected Circle", image_bgr)
            # Wait for 500 ms before moving to next image
            cv2.waitKey(100)
        return None

    circles = np.uint16(np.around(circles[0, :]))
    # For training images, we typically assume there's exactly one chip (or the largest circle is our chip).
    # So let's pick the circle with the largest radius if multiple are found.
    largest_circle = max(circles, key=lambda c: c[2])
    x, y, r = largest_circle

    if show_detection:
        # Draw the detected circle: green for the perimeter and red for the center
        cv2.circle(image_bgr, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image_bgr, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", image_bgr)
        cv2.waitKey(100)

    # Crop a bounding box around the circle
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, image_bgr.shape[1]), min(y + r, image_bgr.shape[0])
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Create a circular mask inside this ROI
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

    # -- FEATURE ENGINEERING: Compute a color histogram in HSV space --
    # Convert to HSV
    hsv = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2HSV)
    # We'll use e.g. 8 bins for H, 8 bins for S, 8 bins for V => total 8*8*8 = 512 features
    h_bins, s_bins, v_bins = 8, 8, 8
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [h_bins, s_bins, v_bins],
                        [0, 180, 0, 256, 0, 256])
    # Normalize the histogram (important for consistent scale)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Flatten to 1D feature vector
    feature_vector = hist.flatten()  # shape (512,)
    return feature_vector


def train_model(dataset_path, model_path="chip_color_model.pkl"):
    """
    1) Walk through the dataset folder structure:
        dataset_path/
            Red/ -> images
            Blue/ -> images
            ...
    2) For each image, detect the circle, extract a color histogram, label with the folder name.
    3) Train a RandomForestClassifier, and save it to disk.

    In training mode, this function now displays each image with the detected circle
    for 500ms so you can verify the detection.
    """
    X = []
    y = []

    # For each subfolder in dataset_path, the folder name is the label
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    if not subfolders:
        print("No subfolders found in dataset path. Please organize your images by color folders.")
        return

    for folder in subfolders:
        label = os.path.basename(folder)  # e.g. "Red", "Green", etc.
        image_files = glob(os.path.join(folder, "*.*"))  # all images in that folder
        print(f"Processing {label} with {len(image_files)} images...")

        for img_file in image_files:
            image_bgr = cv2.imread(img_file)
            if image_bgr is None:
                continue

            # Extract color histogram feature.
            # Set show_detection=True so that the detected circle is displayed.
            feature_vector = extract_chip_roi_and_features(image_bgr, show_detection=True)
            if feature_vector is not None:
                X.append(feature_vector)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("No training data extracted! Check circle detection or dataset images.")
        return

    print(f"Total training samples: {X.shape[0]}")

    # Optional: close the detection display window after processing the dataset
    cv2.destroyAllWindows()

    # Split for an optional train/validation step
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest (you can tweak n_estimators, etc.)
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = clf.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save the model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")



def detect_and_classify_realsense2(model_path="chip_color_model.pkl"):
    """
    Detect poker chips in real-time from a RealSense camera feed,
    process 10 frames (with a 0.2 second delay between them),
    cluster detections that are within 50 pixels of each other,
    and only keep clusters that appear in at least 5 out of the 10 frames.

    Finally, the confirmed stacks are drawn on the original frame in orange
    and displayed until the user presses 'q'.

    Returns:
        confirmed_stacks (list): A list of dictionaries with the details of the confirmed stacks.
    """
    # Load the trained model
    clf = joblib.load(model_path)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920 , 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(1) # To allow the camera to initialize properly and stabilize.

    detections_across_frames = []  # To store detections from all frames
    total_frames = 20
    last_frame = None  # Save the last captured frame to draw on later

    try:
        for frame_idx in range(total_frames):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())  # shape (720, 1280, 3)
            last_frame = frame_bgr.copy()  # Save a copy to annotate later
            frame_depth = np.asanyarray(depth_frame.get_data())  # shape (720, 1280)

            # Create a copy for per-frame annotations (optional)
            annotated_frame = frame_bgr.copy()

            # Detect circles using HoughCircles
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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

            # List for detections in the current frame
            frame_detections = []
            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                # Sort circles by radius descending
                sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
                # Filter overlapping circles: if centers are too close, keep the larger one
                filtered_circles = []
                for circle in sorted_circles:
                    x, y, r = circle
                    keep = True
                    for fc in filtered_circles:
                        fx, fy, fr = fc
                        distance = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                        if distance < 0.5 * (r + fr):
                            keep = False
                            break
                    if keep:
                        filtered_circles.append(circle)

                # Process the filtered circles in the current frame
                for (x, y, r) in filtered_circles:
                    # Draw detected circle for visualization on the current frame
                    cv2.circle(annotated_frame, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), 3)

                    # Define ROI from the circle region
                    x1, y1 = max(0, x - r), max(0, y - r)
                    x2, y2 = min(x + r, frame_bgr.shape[1]), min(y + r, frame_bgr.shape[0])
                    roi = frame_bgr[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    # Create a circular mask for the ROI
                    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (r, r), r, 255, -1)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

                    # Compute HSV histogram features
                    hsv = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8],
                                        [0, 180, 0, 256, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    feature_vector = hist.flatten().reshape(1, -1)

                    # Predict chip color
                    predicted_label = clf.predict(feature_vector)[0]

                    # Retrieve depth information at (x, y)
                    if 0 <= y < frame_depth.shape[0] and 0 <= x < frame_depth.shape[1]:
                        depth = depth_frame.get_distance(x, y)  # in meters
                        depth_mm = depth * 1000  # convert to mm
                    else:
                        depth_mm = None

                    # Save detection details for the current frame
                    detection = {
                        'x': x,
                        'y': y,
                        'depth': depth_mm,
                        'color': predicted_label,
                    }
                    frame_detections.append(detection)

                    # Annotate the frame with prediction text
                    cv2.putText(annotated_frame, predicted_label, (x - 30, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Optionally display the annotated frame for each captured frame
            cv2.imshow("Detection Frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Accumulate detections from this frame
            detections_across_frames.extend(frame_detections)
            time.sleep(0.2)

        # ----- Clustering detections across frames -----
        clusters = []
        distance_thresh = 50  # pixels

        for detection in detections_across_frames:
            x, y = detection['x'], detection['y']
            assigned = False

            # Attempt to add detection into an existing cluster if close enough
            for cluster in clusters:
                cluster_x = np.mean([d['x'] for d in cluster])
                cluster_y = np.mean([d['y'] for d in cluster])
                dist = np.sqrt((x - cluster_x) ** 2 + (y - cluster_y) ** 2)
                if dist <= distance_thresh:
                    cluster.append(detection)
                    assigned = True
                    break

            # Create a new cluster if no match is found
            if not assigned:
                clusters.append([detection])

        # ----- Confirm clusters that have detections in at least 5 out of 10 frames -----
        confirmed_stacks = []
        min_detections = 5

        for cluster in clusters:
            if len(cluster) >= min_detections:
                # Average cluster coordinates and depth
                avg_x = int(np.mean([d['x'] for d in cluster]))
                avg_y = int(np.mean([d['y'] for d in cluster]))
                depths = [d['depth'] for d in cluster if d['depth'] is not None]
                avg_depth = np.mean(depths) if depths else None

                # Determine majority predicted chip color
                colors = [d['color'] for d in cluster]
                majority_color = max(set(colors), key=colors.count)

                stack_info = {
                    'coordinates': (avg_x, avg_y, avg_depth),
                    'color': majority_color,
                    'detections': len(cluster)
                }
                confirmed_stacks.append(stack_info)

        # Print the confirmed stacks information
        if confirmed_stacks:
            print("Confirmed Poker Chip Stacks (detected in >=5/10 frames):")
            for idx, stack in enumerate(confirmed_stacks, start=1):
                x, y, depth = stack['coordinates']
                color = stack['color']
                depth_str = f"{depth:.2f} mm" if depth is not None else "N/A"
                print(
                    f"  Stack {idx}: (x={x}, y={y}, depth={depth_str}), Color: {color}, Detections: {stack['detections']}")
        else:
            print("No poker chip stacks confirmed (detected in at least 5 out of 10 frames).")

        # ----- Draw confirmed stacks on the original (last captured) frame in orange -----
        # Define orange in BGR (e.g., (0, 165, 255))
        display_frame = last_frame.copy() if last_frame is not None else None
        if display_frame is not None:
            for stack in confirmed_stacks:
                x, y, _ = stack['coordinates']
                # Draw an orange circle. Adjust radius as needed (here, 40 pixels)
                cv2.circle(display_frame, (x, y), 40, (0, 165, 255), 2)
                cv2.putText(display_frame, stack['color'], (x - 30, y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Continuously display the final annotated frame until 'q' is pressed.
            while True:
                cv2.imshow("Final Confirmed Stacks", display_frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

        # Optionally wait a short while before finishing if needed
        cv2.waitKey(2000)
        return confirmed_stacks

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "detect"], required=True,
                        help="train or detect mode")
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Path to the dataset for training (only used if mode=train)")
    parser.add_argument("--model_path", type=str, default="chip_color_model.pkl",
                        help="Where to save/load the trained model")

    args = parser.parse_args()

    if args.mode == "train":
        print("Training mode selected.")
        train_model(args.dataset, args.model_path)
    elif args.mode == "detect":
        print("Detection (inference) mode selected.")
        detect_and_classify_realsense2(args.model_path)


if __name__ == "__main__":
    main()
