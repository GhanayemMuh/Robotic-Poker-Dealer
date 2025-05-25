# Poker Dealer Robotic Arm

An automated poker dealer system that uses computer vision, machine learning, and robotics to detect playing cards, classify chip stacks, and manage poker gameplay with minimal human intervention. This project integrates state-of-the-art techniques including YOLOv8-based card recognition, color histogram features with a Random Forest chip classifier, and hand-eye calibration using a ChArUco board to control a robotic arm that deals cards and distributes chips in real time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation & Dependencies](#installation--dependencies)
- [Calibration Process](#calibration-process)
- [Screenshots & Demo](#screenshots--demo)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project aims to revolutionize the traditional poker game experience by automating the dealing process. A robotic arm, calibrated with computer vision, detects playing cards and poker chips on a table, recognizes card ranks and suits via an AI-powered inference engine, and manages chip stacks based on the game state. The system uses:

- **Card Detection:** Leveraging OpenCV’s contour analysis to detect card positions.
- **Card Inference:** Utilizing a YOLOv8-powered inference SDK to classify cards.
- **Chip Classification:** Using a Random Forest model based on HSV histogram features to determine chip colors.
- **Robotic Arm Navigation:** An “eye in hand” configuration calibrated with a ChArUco board to precisely map the camera frame to the robot’s end-effector.
- **Poker Logic:** Evaluating poker hands with the help of the Treys library to decide the winner.

## Features

- **Real-time Card Detection:** Uses contour analysis to detect and track card positions.
- **AI-based Card Recognition:** Processes cropped images of cards to infer their rank and suit.
- **Chip Detection & Classification:** Detects circular regions corresponding to poker chips and classifies them based on color features.
- **Robust Calibration:** Calibrates the camera and robotic arm with a ChArUco board to obtain transformation matrices for accurate movement.
- **Automated Chip Distribution:** Dynamically identifies chip stacks in various table regions (pot, player areas) and directs the robotic arm to distribute chips.
- **Winner Determination:** Converts detected cards to a standard format and evaluates poker hands to declare the winner.
- **Modular Design:** Each component (calibration, detection, inference, chip classification, game logic) is encapsulated in its own module for clarity and maintainability.

## System Architecture

The system is structured into several key modules that communicate through well-defined interfaces:

- **Calibration Module (`calibration.py`):**  
  - Calibrates the camera-to-end-effector transformation using a ChArUco board.  
  - Saves calibration results (intrinsics, distortion coefficients, transformation matrices) for later use.

- **Card Detection & Handling:**  
  - **Card Class (`card.py`):** Defines a card object with position, symbol, and owner attributes. Contains logic to classify cards into “community,” “player1,” or “player2.”
  - **Card Detector (`card_detector.py`):** Uses OpenCV contour detection to locate cards and compute their center coordinates.
  - **Inference Module (`inference.py`):** Preprocesses detected card images and sends them to an external inference API (YOLOv8 model) for classification.

- **Chip Detection & Classification:**  
  - **Chip Classifier (`chip_classifier.py`):** Provides functions to train a Random Forest model using HSV histogram features extracted from detected chip regions. Also includes real-time detection routines using a RealSense camera.
  - **Poker Chips Module (`poker_chips.py`):** Integrates chip detection and classification to detect and confirm chip stacks from various table regions and coordinates with the robotic arm for chip handling.

- **Poker Logic (`poker_logic.py`):**  
  - Implements hand evaluation by converting detected card symbols into a standard format (using the Treys library) and deciding the winning hand.

- **Main Application (`main.py`):**  
  - Acts as the entry point for the system, tying together camera feed updates, card detection, robotic arm navigation, card inference, chip detection, and game logic.  
  - Continuously monitors for new cards, navigates the robotic arm for scanning, and once both players’ cards and community cards are detected, it evaluates the winner and triggers the chip distribution process.

## Installation & Dependencies

Ensure you have **Python 3.7+** installed. Then install the required packages:

```shell
pip install opencv-python numpy scipy scikit-learn joblib pyrealsense2 Pillow treys inference_sdk
```

## Calibration Process

The calibration module (`calibration.py`) uses a ChArUco board to:
- Detect ArUco markers in a series of images.
- Compute intrinsic camera parameters and distortion coefficients.
- Estimate the transformation from the camera to the robot’s end-effector using hand-eye calibration.

Calibration results are saved to `hand_eye_calibration_results.npz` for future runs.

## Screenshots & Demo

### Screenshots

- **Card Detection**  
<table align="center">
  <tr>
    <td align="center">
      <img src="https://i.imghippo.com/files/AhZ3140JA.png" alt="First Image" width="300"/><br>
      <b>Detected 2 of Diamonds with Confidence of 99%</b>
    </td>
    <td align="center">
      <img src="https://i.imghippo.com/files/Kpe7957MCY.png" alt="Second Image" width="300"/><br>
      <b>Detected Ace of Hearts with confidence of 84% </b>
    </td>
    <td align="center">
      <img src="https://i.imghippo.com/files/BHnE1288U.png" alt="Third Image" width="300"/><br>
      <b>Detected Four of Clubs with confidence of 98%</b>
    </td>
  </tr>
</table>




- **Chip Classification**  
  <table align="center">
  <tr>
    <td align="center">
      <img src="https://i.imghippo.com/files/Jbmz6014EbM.png" alt="First Image" width="300"/><br>
      <b>Classification result of 19th Frame.</b>
    </td>
    <td align="center">
      <img src="https://i.imghippo.com/files/kcLP4518kA.png" alt="Second Image" width="300"/><br>
      <b>Classification result of 20th Frame. </b>
    </td>
    <td align="center">
      <img src="https://i.imghippo.com/files/JHQ6355HU.png" alt="Third Image" width="300"/><br>
      <b>Final Decision based off majority.</b>
    </td>
  </tr>
</table>

### Demo Video

[![Watch the video](https://github.com/GhanayemMuh/Poker-Dealer/blob/main/Poker_Dealer_2.gif?raw=true)](https://youtu.be/VeNRNEjk8Dc?si=7AklXDwxeT7V4UBy)

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss your ideas.


## License

This project is licensed under the **MIT License**.

## Contact

For questions or suggestions, please contact:

- **Muhammed Ghanayem** – [GitHub Profile](https://github.com/GhanayemMuh)
- **Ahmad Ghanayem** – [GitHub Profile](https://github.com/ahmadgh99)
- **Supervisor:** Elisei Shafer
