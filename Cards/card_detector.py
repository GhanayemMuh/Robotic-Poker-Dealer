import threading
import time

import cv2
import numpy as np

def show_debug_frame(frame):
    """
    Display a still frame in a separate thread.
    """
    cv2.imshow("Debug Frame", frame)
    print("Press any key to close the still frame...")
    cv2.destroyWindow("Debug Frame")  # Close only the debug window

def detect_objects_from_frame(camera, debug, max_contour_area=60000):
    """
    Detect cards from a frame and return their center coordinates.

    Args:
        frame (numpy.ndarray): The input image frame.
        debug (bool): Whether to add debug information to the frame.
        max_contour_area (int): Maximum allowed contour area for filtering.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of detected card centers.
    """
    # Convert the frame to grayscale
    frame = camera.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > max_contour_area:
            continue  # Filter out contours that are too small or too large

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # Check for rectangular contours
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                card_centers.append((cX, cY))
                if debug:
                    cv2.drawContours(camera.debug_frame, [approx], -1, (0, 255, 0), 2)
                    cv2.circle(camera.debug_frame, (cX, cY), 5, (255, 0, 0), -1)

    if debug:
        # Launch a thread to display the still frame
        debug_thread = threading.Thread(target=show_debug_frame, args=(camera.debug_frame,))
        debug_thread.start()  # Start the thread
        debug_thread.join()   # Optionally wait for the debug window to close

    return card_centers

def detect_cards(frame, debug=False):
    """
    Interface function for external modules to detect cards from a frame.

    Args:
        frame (numpy.ndarray): The input image frame from the camera.
        debug (bool): Whether to add debug information to the frame.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of detected card centers.
    """
    return detect_objects_from_frame(frame, debug)
