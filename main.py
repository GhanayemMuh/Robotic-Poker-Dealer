import threading
import time
import os

import cv2

import inference
import poker_chips
from camera import Camera
import calibration
from card_detector import detect_cards
from arm_controller import Arm
from card import Card
from poker_logic import WinnerDecider
from poker_chips import PokerChips

# Define the path to the calibration file
CALIBRATION_FILE_PATH = "hand_eye_calibration_results.npz"

# Store previously scanned card centers
detected_cards = []
player1_cards = []
player2_cards = []
debug = True
game_over = False
winner = ""

# Global flag to track when cards are being flipped
is_flipping_cards = False

# List to store community cards
community_cards = []

def camera_update(camera, stop_event):
    global debug
    """Thread function to continuously update and display camera frames."""
    try:
        while not stop_event.is_set():
            # Capture RGB frame
            rgb_frame = camera.get_rgb_frame()

            # Ensure the frame is valid before displaying
            if rgb_frame is not None:
                # Display the frame
                cv2.imshow("Live Video Feed", rgb_frame)

                if debug:
                    cv2.imshow("Debug Video Feed", camera.debug_frame)
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            # Add a small delay to control frame rate
            time.sleep(1 / 30)  # Adjust for frame rate (30 FPS)

    except KeyboardInterrupt:
        print("Stopping camera update thread")
    except RuntimeError as e:
        print(f"Camera update error: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        camera.stop()

def check_calibration(camera):
    """Ensure the camera is calibrated to the robotic arm."""
    if not os.path.exists(CALIBRATION_FILE_PATH):
        print("Calibration file not found. Starting calibration process...")
        calibration.calibrate_camera_to_ee()
        camera.load_intrinsics()
        print("Calibration completed and saved.")
    else:
        print("Calibration file found. Skipping calibration.")

def is_new_card(center, existing_cards, threshold=100):
    """Check if a detected card is new based on its center coordinates."""
    for ex_x, ex_y in existing_cards:
        distance = ((center[0] - ex_x)**2 + (center[1] - ex_y)**2)**0.5
        if distance < threshold:
            return False
    return True

def is_new_label(label, existing_cards):
    """Check if a card with the given label is already in the list."""
    return label not in [card.symbol for card in existing_cards]

def handle_card_detection(camera, arm):
    global detected_cards, is_flipping_cards, winner, game_over, community_cards, player1_cards, player2_cards

    if len(community_cards) == 5:
        print("all community cards detected.")

    if len(player1_cards) == 2:
        print("player 1 cards detected.")

    if len(player2_cards) == 2:
        print("player 2 cards detected.")

    # Reset the flipping flag if no hand is detected
    is_flipping_cards = False

    rgb_frame = camera.get_rgb_frame()
    card_centers = detect_cards(camera, debug)
    card_centers.sort(key=lambda card: (card[0], card[1]), reverse=True)
    depth_frame = camera.get_depth_frame()

    time.sleep(0.5)

    for center in card_centers:
        if is_new_card(center, [card.position for card in detected_cards]):
            # Create a new card object
            card = Card(position=center)
            Card.classify_cards(card)  # Determine card ownership (community, player1, or player2)

            # Navigate to and scan the new card
            print(f"New card detected at {center}. Navigating...")
            arm.navigate_to_card(center[0], center[1], depth_frame, camera, scan_flag=True)
            time.sleep(0.2)

            # Perform inference
            consolidated_results = inference.stable_inference(camera, debug)

            # Filter results with confidence >= 0.8 and classify cards
            for result in consolidated_results:
                label = result["label"]
                confidence = result["confidence"]

                if confidence >= 0.80:
                    # Check ownership and add to the appropriate list
                    if card.owner == "community" and is_new_label(label, community_cards):
                        print(f"Adding community card {label} with confidence {confidence}")
                        card.symbol = label
                        community_cards.append(Card(position=center, symbol=label))
                    elif card.owner == "player1" and is_new_label(label, player1_cards):
                        print(f"Adding Player 1 card {label} with confidence {confidence}")
                        card.symbol = label
                        player1_cards.append(Card(position=center, symbol=label))
                    elif card.owner == "player2" and is_new_label(label, player2_cards):
                        print(f"Adding Player 2 card {label} with confidence {confidence}")
                        card.symbol = label
                        player2_cards.append(Card(position=center, symbol=label))

            # Add the card to the detected list to avoid reprocessing
            detected_cards.append(card)

    # Check if the game should end
    if len(player1_cards) == 2 and len(player2_cards) == 2:
        print("All non-community cards detected. Game ends.")
        winner =  WinnerDecider().decide_winner(player1_cards, player2_cards, community_cards)
        print(f"Winner is {winner}")
        game_over = True
        return winner

    arm.return_to_natural()
    return None

def clear_cards(camera,arm):
    global detected_cards

    depth_frame = camera.get_depth_frame()
    
    for card in detected_cards:
        arm.navigate_to_card(card.position[0], card.position[1], depth_frame, camera, scan_flag=False)
        arm.throw_cards()


def main():
    """Main entry point for the application."""

    global debug

    # Initialize the camera
    camera = Camera()
    # Check if calibration is needed
    check_calibration(camera)
    # Initialize arm
    arm = Arm()
    arm.return_to_natural()

    # Capture initial baseline depth in ROI
    camera.capture_baseline_depth(roi_rgb=(500, 700, 1980, 1080))

    # Start a thread to update the camera
    stop_event = threading.Event()
    camera_thread = threading.Thread(target=camera_update, args=(camera, stop_event), daemon=True)
    camera_thread.start()
    time.sleep(0.5)

    try:
        while True:
            # Call card detection periodically
            winner = handle_card_detection(camera, arm)
            time.sleep(2)  # Adjust interval as needed
            #winner = "player1"
            #game_over = True
            if game_over:
                poker_chips = PokerChips()
                winner_stacks = poker_chips.detect_and_classify_stacks(camera, arm, region = winner,debug=debug)
                pot_stacks = poker_chips.detect_and_classify_stacks(camera, arm, "pot",debug=debug)
                arm.deliver_poker_chips(winner_stacks, pot_stacks, winner)
                arm.return_to_natural()
                clear_cards(camera,arm)
                break

    except KeyboardInterrupt:
        print("Shutting down application...")
    finally:
        stop_event.set()  # Signal the camera thread to stop
        camera_thread.join()
        camera.stop()

if __name__ == "__main__":
    main()
