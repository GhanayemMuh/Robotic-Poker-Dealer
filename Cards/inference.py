import threading
import time

import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import os
from collections import Counter

# Set a custom temporary directory where you have write permissions
temp_dir = os.path.expanduser(r'~\Downloads\temp')
os.makedirs(temp_dir, exist_ok=True)

# Set up the Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="wY5Env61qpo1Xh6Gc9pN"
)

def show_debug_frame(frame):
    """
    Display a still frame in a separate thread.
    """
    cv2.imshow("Debug Frame", frame)
    print("Press any key to close the still frame...")
    cv2.waitKey(1000)
    #time.sleep(1)
    cv2.destroyWindow("Debug Frame")  # Close only the debug window

def preprocess_cropped_image(cropped_frame, output_size=(512, 288)):
    """
    Preprocess the cropped image by adding a black background and centering the cropped frame.

    Args:
        cropped_frame (numpy.ndarray): The cropped frame of the detected contour.
        output_size (tuple): The desired output size (width, height).

    Returns:
        numpy.ndarray: The preprocessed image with a black background.
    """
    # Get dimensions of the cropped image
    h, w = cropped_frame.shape[:2]

    # Create a black background of the specified size
    target_w, target_h = output_size
    black_background = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Resize the cropped image to fit within the black background while maintaining aspect ratio
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    if aspect_ratio > target_aspect_ratio:
        resized_w = target_w
        resized_h = int(target_w / aspect_ratio)
    else:
        resized_h = target_h
        resized_w = int(target_h * aspect_ratio)

    resized_cropped = cv2.resize(cropped_frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Calculate center offset for placing the resized image in the black background
    x_offset = (target_w - resized_w) // 2
    y_offset = (target_h - resized_h) // 2

    # Place the resized image in the center of the black background
    black_background[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized_cropped

    return black_background

def stable_inference(camera, debug=False):
    results = []
    last_frame = None

    for _ in range(1):
        frame = camera.frame  # Obtain the frame from the background thread
        if frame is None:
            raise RuntimeError("No frame available from the camera.")

        # Ensure the frame is different from the last one to avoid duplicates
        if np.array_equal(frame, last_frame):
            continue
        last_frame = frame

        # Convert the frame to grayscale for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_results = []

        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the polygon is roughly rectangular (4 sides)
            if len(approx) == 4 and cv2.contourArea(approx) > 10000:  # Ignore small contours
                # Create a bounding box for the polygon
                x, y, w, h = cv2.boundingRect(approx)
                cropped_frame = frame[y:y + h, x:x + w]

                # Preprocess the cropped frame to add a black background
                preprocessed_image = preprocess_cropped_image(cropped_frame)

                # Save the preprocessed image for inference
                pil_image = Image.fromarray(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
                temp_file_path = os.path.join(temp_dir, "temp_image.jpg")
                pil_image.save(temp_file_path)

                # Perform inference on the preprocessed image
                result = CLIENT.infer(
                    temp_file_path,
                    model_id="playing-cards-u9csc/1"
                )

                # Extract the label with the highest confidence
                predictions = result.get("predictions", [])
                if predictions:
                    best_prediction = max(predictions, key=lambda x: x["confidence"])
                    label = best_prediction["class"]
                    confidence = best_prediction["confidence"]

                    filtered_results.append({"label": label, "confidence": confidence})

                    # Debugging: Annotate the preprocessed image with the best label
                    if debug:
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(preprocessed_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Display the preprocessed image
                        debug_thread = threading.Thread(target=show_debug_frame, args=(preprocessed_image,))
                        debug_thread.start()
                        debug_thread.join()

        results.append(filtered_results)

    if not results:
        return []

    # Consolidate results from all iterations
    consolidated_results = [res for result_set in results for res in result_set]
    return consolidated_results
