import cv2
import mediapipe as mp
import math
import time

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

cap = cv2.VideoCapture(0)
pTime = 0

# Drawing Points
mpDraw = mp.solutions.drawing_utils
# Pose Detection
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Prompt the user to stand 5 feet away from the camera
print("Please stand 5 feet away from the camera and ensure your entire body is visible.")

while True:
    success, img = cap.read()
    # Conversion because mediapipe works on RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # Draw pose landmarks on the image
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))

        # Body part IDs for measurements
        waist_id = 23
        head_id = 0
        left_arm_id = 11
        right_arm_id = 12
        left_leg_id = 23
        right_leg_id = 24

        # Check if all required landmarks are detected
        if all(results.pose_landmarks.landmark[id].visibility >= 0.5 for id in [waist_id, head_id, left_arm_id, right_arm_id, left_leg_id, right_leg_id]):
            # Get landmark positions
            landmarks = results.pose_landmarks.landmark

            # Get image dimensions
            h, w, c = img.shape

            # Calculate distances for body measurements
            waist_distance = calculate_distance(landmarks[waist_id].x * w, landmarks[waist_id].y * h,
                                                landmarks[head_id].x * w, landmarks[head_id].y * h)
            head_to_feet_distance = calculate_distance(landmarks[head_id].x * w, landmarks[head_id].y * h,
                                                       landmarks[left_leg_id].x * w, landmarks[left_leg_id].y * h)

            # Convert distances to meters
            pixel_to_meter = head_to_feet_distance / 4.0
            waist_distance_m = round(waist_distance / pixel_to_meter, 2)
            height_m = round(head_to_feet_distance / pixel_to_meter, 2)

            # Calculate other body measurements using waist_distance as a reference
            left_arm_distance_m = round(waist_distance_m * 0.22, 2)
            right_arm_distance_m = round(waist_distance_m * 0.22, 2)
            left_leg_distance_m = round(waist_distance_m * 0.48, 2)
            right_leg_distance_m = round(waist_distance_m * 0.48, 2)
            chest_distance_m = round(waist_distance_m * 0.70, 2)

            # Display measurements on the image
            cv2.putText(img, f"Waist: {waist_distance_m} m", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Left Arm: {left_arm_distance_m} m", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Right Arm: {right_arm_distance_m} m", (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Left Leg: {left_leg_distance_m} m", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Right Leg: {right_leg_distance_m} m", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Chest: {chest_distance_m} m", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            cv2.putText(img, f"Height: {height_m} m", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
        else:
            # Not all required body parts detected
            cv2.putText(img, "Please ensure all body parts are visible", (70, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    else:
        # No pose landmarks detected
        cv2.putText(img, "No pose landmarks detected", (70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

