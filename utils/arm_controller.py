import numpy as np
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
import time

class Arm:
    def __init__(self):
        # Initialize xArm
        self.arm = XArmAPI('192.168.1.170')  # Replace with your xArm's IP address
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

        # Load calibration data
        calibration_data = np.load("hand_eye_calibration_results.npz")
        self.T_cam2ee = calibration_data["T_cam2ee"]  # Transformation matrix from camera to end-effector
        self.T_cam2ee[0][3] += 0.03 # X offset in ee frame.
        self.T_cam2ee[1][3] -= 0.012 # Y offset in ee frame
        self.natural_pose_transform_matrix = None

    @staticmethod
    def euler_to_transformation_matrix(position):
        """
        Convert xArm position [x, y, z, roll, pitch, yaw] to a 4x4 transformation matrix.
        """
        x, y, z, roll, pitch, yaw = position
        rotation_matrix = R.from_euler('zyx', [-yaw, pitch, roll], degrees=False).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]  # Convert mm to meters
        return transformation_matrix

    def navigate_to_card(self, x_pixel, y_pixel, depth_frame, camera, scan_flag=False):
        """
        Navigate the robotic arm to a specific (x, y) pixel coordinate detected in the camera frame.

        :param x_pixel: X pixel coordinate.
        :param y_pixel: Y pixel coordinate.
        :param depth_frame: The depth frame from the camera.
        :param camera: Camera.
        :param scan_flag: Flag indicating whether to adjust for scanning offset.
        """
        # Extract a 6x6 depth region around the pixel
        depth_array = np.asanyarray(depth_frame)
        h, w = depth_array.shape
        x_min, x_max = max(0, x_pixel - 3), min(w, x_pixel + 3)
        y_min, y_max = max(0, y_pixel - 3), min(h, y_pixel + 3)
        depth_region = depth_array[y_min:y_max, x_min:x_max]

        # Calculate the average depth value, ignoring zeros
        valid_depths = depth_region[depth_region > 0]
        if valid_depths.size == 0:
            print(f"No valid depth values around pixel ({x_pixel}, {y_pixel}).")
            return
        depth = np.mean(valid_depths) * camera.depth_scale
        depth_intrinsics = camera.depth_frame.profile.as_video_stream_profile().get_intrinsics()

        # Deproject pixel to 3D in the camera frame
        point_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel], depth)

        # Get the current pose of the end-effector in the base frame
        code, current_pose = self.arm.get_position(is_radian=True)  # Get [x, y, z, roll, pitch, yaw]
        if code != 0:
            print("Failed to get end-effector position.")
            return
        T_base2ee = self.natural_pose_transform_matrix

        # Transform the target point to the robot's base frame
        target_position_ee_frame = np.dot(self.T_cam2ee[:3, :3], point_camera) + self.T_cam2ee[:3, 3]
        target_position_ee_homogeneous = np.append(target_position_ee_frame, 1)
        target_position_base_frame = np.dot(T_base2ee, target_position_ee_homogeneous)[:3]

        print(f"Target position in base frame: {target_position_base_frame}")

        # Command the robot to move to the detected position
        x, y, z = target_position_base_frame
        if scan_flag:
            pose = [(x * 1000) - 40, y * 1000, 250, -180, 0, 0]
        else:
            pose = [x * 1000, y * 1000, 0, -180, 0, 0]

        #_, angles = self.arm.get_inverse_kinematics(pose, False, False)
        #if angles is not None:
        self.arm.set_position(x=pose[0],y=pose[1],z=pose[2],pitch=0,roll=-180,yaw=0, speed=220, mvacc=400, wait=True)
        time.sleep(0.5)
        #else:
        #    print("Failed to compute inverse kinematics for the target pose.")

    def return_to_natural(self):
        """
        Move the robot to a predefined "natural" position in the base frame.
        The natural position is defined as:
        - X = 200 mm
        - Y = 0 mm
        - Z = 450 mm
        - Roll = 0 rad
        - Pitch = 0 rad
        - Yaw = 0 rad
        """
        natural_position = [180, 0, 450, 180, 0, 0]  # x, y, z in mm, roll, pitch, yaw in radians
        x, y, z, roll, pitch, yaw = natural_position

        # Deactivate the vacuum gripper
        self.arm.set_vacuum_gripper(False)

        # Command the robot to move to the natural position
        success = self.arm.set_position(x, y, z, roll, pitch, yaw, speed=220, mvacc=400, wait=True)

        code, current_pose = self.arm.get_position(is_radian=True)
        if code == 0:
            self.natural_pose_transform_matrix = self.euler_to_transformation_matrix(current_pose)
        if success != 0:
            raise RuntimeError(f"Failed to move to natural position. Error code: {success}")

        print(f"Robot moved to natural position: X={x} mm, Y={y} mm, Z={z} mm")

    def move_to_stacks(self, region):
        if region == "player1":
            self.arm.set_position(220, -200, 300, -180, 0, -90, wait=True)

        elif region == "player2":
            self.arm.set_position(220, 150, 300 , -180, 0, 90, wait=True)

        elif region == "pot":
            self.arm.set_position(160, 0, 300, -180, 0, 0, wait=True)

        else:
            return

    def deliver_poker_chips(self, winner_stacks, pot_stacks, region):
        """
        Move stacks of chips from the pot to the winner stacks.
        Adjust yaw for winner stacks based on the region.
        If a winner has no stacks of a given color, a new stack is created
        at the lowest X-position stack's Y-coordinate with a 1 cm shift in X.
        """
        CURRENT_THRESHOLD = 0.0  # Threshold for Z-axis force in N

        # Determine yaw based on the region
        if region == "player1":
            winner_yaw = -90
        elif region == "player2":
            winner_yaw = -90
        else:  # Default to no yaw for pot
            winner_yaw = 0

        self.arm.set_collision_sensitivity(1)  # Enable collision detection

        # Step 1: Determine the lowest X-position of winner stacks
        if winner_stacks:
            min_x_stack = min(winner_stacks, key=lambda stack: stack['center'][0])  # Stack with lowest X
            min_x = min_x_stack['center'][0]
            min_x_y = min_x_stack['center'][1]  # Use its Y as reference
        else:
            min_x = None  # No stacks exist yet
            min_x_y = None

        # Step 2: Process pot stacks
        for pot_stack in pot_stacks:
            pot_color = pot_stack['color']

            # Check if the winner already has a stack of this color
            matching_winner_stack = next((stack for stack in winner_stacks if stack['color'] == pot_color), None)

            # If no matching stack is found, create a new stack location
            if matching_winner_stack is None:
                if min_x is not None:
                    min_x -= 0.060  # Shift 1 cm left for each new stack
                else:
                    min_x = pot_stack['center'][0]  # Default to pot stack x if no winner stacks exist

                new_stack_x = min_x
                new_stack_y = min_x_y if min_x_y is not None else pot_stack['center'][
                    1]  # Use lowest X stack's Y, fallback to pot's Y
                new_stack_z = pot_stack['center'][2]  # Start at the same Z

                # Add new stack to winner stacks list and UPDATE min_x
                matching_winner_stack = {
                    'center': (new_stack_x, new_stack_y, new_stack_z),
                    'color': pot_color
                }
                winner_stacks.append(matching_winner_stack)

            # Initialize Z-coordinates
            current_winner_z = matching_winner_stack['center'][2] * 1000 + 5  # Winner stack Z in mm
            current_pot_z = pot_stack['center'][2] * 1000 + 5  # Pot stack Z in mm

            last_attempt = False

            # Step 3: Transfer chips from pot stack to the corresponding winner stack
            while current_pot_z > 0:  # Continue until the pot stack is depleted
                # Move above the pot stack (yaw = 0 for pot)
                self.arm.set_position(
                    x=pot_stack['center'][0] * 1000,
                    y=pot_stack['center'][1] * 1000,
                    z=current_pot_z,
                    roll=-180,
                    pitch=0,
                    yaw=0,  # No yaw adjustment for pot
                    speed=220,
                    mvacc=400,
                    wait=True
                )

                # Activate the vacuum gripper
                self.arm.set_vacuum_gripper(True)

                # Move down to pick up the chip
                while True:
                    code, current_pose = self.arm.get_position(is_radian=True)
                    if code != 0:
                        print("Failed to get end-effector position.")
                        return

                    # Decrease Z incrementally
                    new_z = current_pose[2] - 1
                    if new_z < 3 and not last_attempt:
                        print(
                            f"Reached near lower limit for pot stack at {pot_stack['center']}. This will be the last attempt.")
                        last_attempt = True

                    if new_z < 0:  # Safety limit
                        print("Reached lower limit for pot stack. Moving to the next stack.")
                        self.arm.set_vacuum_gripper(False)
                        break

                    self.arm.set_position(
                        x=current_pose[0],
                        y=current_pose[1],
                        z=new_z,
                        roll=-180,
                        pitch=0,
                        yaw=0,  # No yaw adjustment for pot
                        speed=10,
                        mvacc=400
                    )

                    if self.arm.get_vacuum_gripper()[1] == 1:
                        print(f"Picked up chip from pot stack at {pot_stack['center']}.")
                        break

                    time.sleep(0.2)

                # Lift the chip
                self.arm.set_position(
                    x=current_pose[0],
                    y=current_pose[1],
                    z=100,  # Lift height
                    roll=-180,
                    pitch=0,
                    yaw=0,  # No yaw adjustment for pot
                    speed=220,
                    mvacc=400,
                    wait=True
                )

                # Move above the winner stack (apply winner_yaw)
                self.arm.set_position(
                    x=matching_winner_stack['center'][0] * 1000,
                    y=matching_winner_stack['center'][1] * 1000,
                    z=current_winner_z,  # Approach height
                    roll=-180,
                    pitch=0,
                    yaw=winner_yaw,  # Apply yaw adjustment for winner stack
                    speed=220,
                    mvacc=400,
                    wait=True
                )

                # Move down to place the chip
                while True:
                    joint_currents = self.arm.currents  # Read joint currents
                    if joint_currents and joint_currents[1] > CURRENT_THRESHOLD:
                        print("Detected resistance or collision during placement.")
                        self.arm.set_vacuum_gripper(False)
                        break

                    code, current_pose = self.arm.get_position(is_radian=True)
                    if code != 0:
                        print("Failed to get position.")
                        return

                    # Decrease Z incrementally
                    new_z = current_pose[2] - 1
                    if new_z < 0:  # Safety limit
                        print("Reached lower limit.")
                        self.arm.set_vacuum_gripper(False)
                        break

                    self.arm.set_position(
                        x=current_pose[0],
                        y=current_pose[1],
                        z=new_z,
                        roll=-180,
                        pitch=0,
                        yaw=winner_yaw,  # Maintain yaw adjustment
                        speed=10,
                        mvacc=400
                    )

                    time.sleep(0.2)

                # Lift the arm after placing the chip
                self.arm.set_position(
                    x=current_pose[0],
                    y=current_pose[1],
                    z=100,  # Lift height
                    roll=-180,
                    pitch=0,
                    yaw=winner_yaw,  # Maintain yaw adjustment
                    speed=220,
                    mvacc=400,
                    wait=True
                )

                print(f"Delivered chip to winner stack at {matching_winner_stack['center']}.")

                # Decrement the Z-coordinate for the pot stack
                current_pot_z -= 3
                current_winner_z += 3
                # If this was the last attempt, move to the next pot stack
                if last_attempt:
                    print(f"Finished processing pot stack at {pot_stack['center']}.")
                    break

        print("All pot stacks have been processed and delivered to the winner stacks.")

    def throw_cards(self):
        self.arm.set_vacuum_gripper(True)

        time.sleep(0.3)

        self.arm.set_position(x=0,y=0,z=30,pitch=0,yaw=0,roll=0,speed=220,mvacc=400,relative=True,wait=True)

        self.arm.set_position(x=417,
                              y=0,
                              z=0,
                              roll=-180,
                              pitch=0,
                              yaw=0,
                              speed=220,
                              mvacc=400,
                              wait=True)

        self.arm.set_vacuum_gripper(False)

        time.sleep(0.3)
        self.arm.set_position(x=0,y=0,z=30,pitch=0,yaw=0,roll=0,speed=220,mvacc=400,relative=True,wait=True)


        self.arm.set_position(x=202,
                    y=0,
                    z=130,
                    roll=-180,
                    pitch=0,
                    yaw=-90,
                    speed=220,
                    mvacc=400,
                    wait=True)

