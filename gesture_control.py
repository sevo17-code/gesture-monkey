import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

if not hasattr(mp, "solutions"):
    print("[ERROR] Your installed mediapipe package does not expose 'mp.solutions'.")
    print("[ERROR] This script needs the classic MediaPipe Solutions API.")
    print("[ERROR] Install Python 3.11, then run:")
    print("[ERROR] python -m pip install opencv-python mediapipe==0.10.14 numpy")
    sys.exit(1)

MONKEY_IMAGES = [
    "assets/Monkey1.jpg",  # Hands on head + scream
    "assets/Monkey2.jpg",  # Dab
    "assets/Monkey3.jpg",  # Thumbs up
    "assets/Monkey4.jpg",  # Finger on mouth
    "assets/Monkey5.jpg",  # Peace sign + smile
]

MEME_DISPLAY_SECONDS = 3
SHUTDOWN_CONFIRM = False
SHUTDOWN_HOLD_SECONDS = 2.0

MOUTH_TOUCH_THRESHOLD = 0.06
HEAD_TOUCH_THRESHOLD = 0.18
SCREAM_MOUTH_OPEN_THRESHOLD = 0.10
SMILE_RATIO_THRESHOLD = 0.34
DAB_NOSE_DISTANCE_THRESHOLD = 0.22
POSE_VISIBILITY_THRESHOLD = 0.5

# OpenCV uses BGR.
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (200, 200, 200)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

mp_drawing = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS


def distance(p1, p2):
    """Return Euclidean distance between two normalized points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_handedness_label(handedness_item):
    """Return hand label from MediaPipe handedness result."""
    if handedness_item is None or not handedness_item.classification:
        return "Unknown"
    return handedness_item.classification[0].label


def get_raised_fingers(hand_landmarks, hand_label):
    """Return flags for raised fingers [thumb, index, middle, ring, pinky]."""
    lm = hand_landmarks.landmark
    raised = []

    thumb_tip = (lm[4].x, lm[4].y)
    thumb_ip = (lm[3].x, lm[3].y)
    if hand_label == "Right":
        raised.append(thumb_tip[0] < thumb_ip[0])
    elif hand_label == "Left":
        raised.append(thumb_tip[0] > thumb_ip[0])
    else:
        raised.append(abs(thumb_tip[0] - thumb_ip[0]) > 0.03)

    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        raised.append(lm[tip_id].y < lm[pip_id].y)

    return raised


def is_middle_finger(raised_fingers):
    """Return True when only middle finger is raised."""
    if len(raised_fingers) < 5:
        return False
    _, index, middle, ring, pinky = raised_fingers
    return middle and not index and not ring and not pinky


def is_peace_sign(raised_fingers):
    """Return True when index + middle are up and ring + pinky are down."""
    if len(raised_fingers) < 5:
        return False
    _, index, middle, ring, pinky = raised_fingers
    return index and middle and not ring and not pinky


def is_thumbs_up(hand_landmarks, raised_fingers):
    """Return True for thumbs-up style hand pose."""
    if len(raised_fingers) < 5:
        return False

    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    thumb_up = thumb_tip.y < thumb_mcp.y
    other_fingers_down = not any(raised_fingers[1:])
    return thumb_up and other_fingers_down


def extract_face_features(face_landmarks):
    """Extract mouth center, mouth-open ratio, and smile ratio."""
    if face_landmarks is None:
        return None

    lm = face_landmarks.landmark
    mouth_center = (lm[13].x, lm[13].y)
    mouth_open = abs(lm[14].y - lm[13].y)
    face_width = distance((lm[33].x, lm[33].y), (lm[263].x, lm[263].y))
    mouth_width = distance((lm[61].x, lm[61].y), (lm[291].x, lm[291].y))

    if face_width <= 1e-6:
        return None

    return {
        "mouth_center": mouth_center,
        "mouth_open_ratio": mouth_open / face_width,
        "smile_ratio": mouth_width / face_width,
    }


def is_index_touching_mouth(hand_landmarks, mouth_center):
    """Return True when index fingertip is close to mouth center."""
    if hand_landmarks is None or mouth_center is None:
        return False

    index_tip = hand_landmarks.landmark[8]
    return distance((index_tip.x, index_tip.y), mouth_center) < MOUTH_TOUCH_THRESHOLD


def is_two_hands_on_head(hand_contexts, face_landmarks):
    """Return True when two hands are near left/right temple regions."""
    if face_landmarks is None or len(hand_contexts) < 2:
        return False

    lm_face = face_landmarks.landmark
    left_temple = (lm_face[127].x, lm_face[127].y)
    right_temple = (lm_face[356].x, lm_face[356].y)

    near_left = False
    near_right = False
    for hand in hand_contexts:
        lm_hand = hand["landmarks"].landmark
        wrist = (lm_hand[0].x, lm_hand[0].y)
        index_tip = (lm_hand[8].x, lm_hand[8].y)
        if min(distance(wrist, left_temple), distance(index_tip, left_temple)) < HEAD_TOUCH_THRESHOLD:
            near_left = True
        if min(distance(wrist, right_temple), distance(index_tip, right_temple)) < HEAD_TOUCH_THRESHOLD:
            near_right = True

    return near_left and near_right


def is_dab_pose(pose_landmarks):
    """Return True for a rough dab pose using body landmarks."""
    if pose_landmarks is None:
        return False

    lm = pose_landmarks.landmark
    required_ids = [0, 11, 12, 13, 14, 15, 16]
    if any(lm[idx].visibility < POSE_VISIBILITY_THRESHOLD for idx in required_ids):
        return False

    nose = (lm[0].x, lm[0].y)

    left_shoulder = (lm[11].x, lm[11].y)
    right_shoulder = (lm[12].x, lm[12].y)
    left_elbow = (lm[13].x, lm[13].y)
    right_elbow = (lm[14].x, lm[14].y)
    left_wrist = (lm[15].x, lm[15].y)
    right_wrist = (lm[16].x, lm[16].y)

    left_arm_up = left_wrist[1] < left_shoulder[1] - 0.08 and left_wrist[0] < left_elbow[0]
    right_arm_up = right_wrist[1] < right_shoulder[1] - 0.08 and right_wrist[0] > right_elbow[0]

    right_arm_cross_face = distance(right_wrist, nose) < DAB_NOSE_DISTANCE_THRESHOLD
    left_arm_cross_face = distance(left_wrist, nose) < DAB_NOSE_DISTANCE_THRESHOLD

    return (left_arm_up and right_arm_cross_face) or (right_arm_up and left_arm_cross_face)


def detect_monkey_pose(hand_contexts, face_landmarks, pose_landmarks):
    """Map live pose to Monkey1..Monkey5."""
    face = extract_face_features(face_landmarks)
    mouth_center = face["mouth_center"] if face else None
    mouth_open_ratio = face["mouth_open_ratio"] if face else 0.0
    smile_ratio = face["smile_ratio"] if face else 0.0

    if face and is_two_hands_on_head(hand_contexts, face_landmarks) and mouth_open_ratio > SCREAM_MOUTH_OPEN_THRESHOLD:
        return "MONKEY_1"

    if is_dab_pose(pose_landmarks):
        return "MONKEY_2"

    if any(is_thumbs_up(hand["landmarks"], hand["raised"]) for hand in hand_contexts):
        return "MONKEY_3"

    if mouth_center and any(is_index_touching_mouth(hand["landmarks"], mouth_center) for hand in hand_contexts):
        return "MONKEY_4"

    if face and smile_ratio > SMILE_RATIO_THRESHOLD and any(is_peace_sign(hand["raised"]) for hand in hand_contexts):
        return "MONKEY_5"

    return None


def load_all_memes(paths, target_size=(640, 480)):
    """Load all monkey images and keep placeholders for missing files."""
    loaded = []
    for index, path in enumerate(paths, start=1):
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                loaded.append(cv2.resize(img, target_size))
                continue

        print(f"[!] Could not load '{path}'. Using placeholder.")
        placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        placeholder[:] = (40, 40, 40)
        cv2.putText(
            placeholder,
            f"Missing Monkey{index}.jpg",
            (30, target_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR_RED,
            2,
        )
        loaded.append(placeholder)

    return loaded


def compose_output_frame(camera_frame, meme_img):
    """Show camera on the left and matched monkey image on the right."""
    if meme_img is None:
        return camera_frame

    frame_h, frame_w = camera_frame.shape[:2]
    panel_w = max(260, frame_w // 2)
    panel = cv2.resize(meme_img, (panel_w, frame_h))
    cv2.putText(panel, "Monkey Match", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, 2)
    return np.hstack((camera_frame, panel))


def get_image_by_index(meme_imgs, image_index):
    """Return image by index with safe fallback."""
    if not meme_imgs:
        return None
    safe_index = min(max(0, image_index), len(meme_imgs) - 1)
    return meme_imgs[safe_index]


def execute_shutdown():
    """Run shutdown command. Return True if shutdown command was executed."""
    if SHUTDOWN_CONFIRM:
        print("\n" + "=" * 40)
        print("Shutdown gesture detected.")
        print("Type 'yes' to confirm or anything else to cancel:")
        confirm = input(">>> ").strip().lower()
        if confirm != "yes":
            print("Shutdown canceled.")
            return False

    print("Shutting down...")
    time.sleep(1)
    if os.name == "nt":
        os.system("shutdown /s /t 0")
    else:
        os.system("sudo shutdown -h now")
    return True


def main():
    """Run live monkey-pose matching."""
    print("Starting monkey pose matcher...")
    print("Match these poses to show image beside you:")
    print("   - Monkey1: both hands on head + open mouth")
    print("   - Monkey2: dab pose")
    print("   - Monkey3: thumbs up")
    print("   - Monkey4: index finger near mouth")
    print("   - Monkey5: peace sign + smile")
    print(f"   - Middle finger (any hand) for {SHUTDOWN_HOLD_SECONDS:.1f}s = shutdown")
    print("   - Press Q to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meme_imgs = load_all_memes(MONKEY_IMAGES, (frame_width, frame_height))

    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    face_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    current_meme_img = None
    meme_end_time = 0
    gesture_cooldown = 1.2
    last_gesture_time = 0
    middle_finger_start_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hand_detector.process(rgb_frame)
            face_results = face_detector.process(rgb_frame)
            pose_results = pose_detector.process(rgb_frame)

            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

            pose_landmarks = pose_results.pose_landmarks if pose_results else None

            display_frame = frame.copy()
            current_time = time.time()
            gesture_detected = None
            hold_remaining = None
            hand_contexts = []

            if hand_results.multi_hand_landmarks:
                handedness_list = hand_results.multi_handedness or []
                for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = "Unknown"
                    if hand_index < len(handedness_list):
                        hand_label = get_handedness_label(handedness_list[hand_index])

                    raised = get_raised_fingers(hand_landmarks, hand_label)
                    hand_contexts.append(
                        {
                            "label": hand_label,
                            "landmarks": hand_landmarks,
                            "raised": raised,
                        }
                    )

                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands_connections,
                        mp_drawing.DrawingSpec(color=COLOR_RED, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=COLOR_BLUE, thickness=2),
                    )

                    wrist = hand_landmarks.landmark[0]
                    cv2.putText(
                        display_frame,
                        hand_label,
                        (int(wrist.x * frame_width), int(wrist.y * frame_height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLOR_LIGHT_GRAY,
                        1,
                    )

            any_middle_finger = any(is_middle_finger(hand["raised"]) for hand in hand_contexts)
            if any_middle_finger:
                if middle_finger_start_time is None:
                    middle_finger_start_time = current_time
                hold_time = current_time - middle_finger_start_time
                hold_remaining = max(0.0, SHUTDOWN_HOLD_SECONDS - hold_time)
                if hold_time >= SHUTDOWN_HOLD_SECONDS:
                    gesture_detected = "SHUTDOWN"
            else:
                middle_finger_start_time = None

            if gesture_detected is None:
                gesture_detected = detect_monkey_pose(hand_contexts, face_landmarks, pose_landmarks)

            if gesture_detected and (current_time - last_gesture_time > gesture_cooldown):
                last_gesture_time = current_time

                if gesture_detected == "SHUTDOWN":
                    print("Middle finger hold confirmed.")
                    if execute_shutdown():
                        return

                elif gesture_detected.startswith("MONKEY_"):
                    monkey_num = int(gesture_detected.split("_")[1])
                    print(f"Matched Monkey{monkey_num}.")
                    current_meme_img = get_image_by_index(meme_imgs, monkey_num - 1)
                    meme_end_time = current_time + MEME_DISPLAY_SECONDS

            side_image = current_meme_img if current_time < meme_end_time else None

            if hold_remaining is not None and hold_remaining > 0:
                status_color = COLOR_RED
                status_text = f"Hold middle finger: {hold_remaining:.1f}s"
            elif gesture_detected:
                status_color = COLOR_GREEN
                status_text = gesture_detected
            else:
                status_color = COLOR_GRAY
                status_text = "Monitoring..."

            cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(
                display_frame,
                "Q = exit",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_LIGHT_GRAY,
                1,
            )

            final_frame = compose_output_frame(display_frame, side_image)
            cv2.imshow("Gesture Control", final_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print(f"[CRITICAL ERROR] Unexpected error: {error}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        face_detector.close()
        pose_detector.close()
        print("Program closed.")


if __name__ == "__main__":
    main()
