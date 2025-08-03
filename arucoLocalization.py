import cv2
import numpy as np

# 1. ArUco Configuration
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # Would it be better / worse to use different dimentions?
MARKER_LENGTH = 0.15  # in meters. TODO: REPLACE WITH ACTUAL MARKER SIZE

# 2. Calibrate Camera 
# TODO: REPLACE WITH ACTUAL CAMERA CALIBRATION CONSTANTS!!!!
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Replace if using lens distortion

# 3. Initialize Video Capture ===
cap = cv2.VideoCapture(1)

# 4. Position Storage
marker_positions = {}      # {marker_id: np.array([x, y, z])}
filtered_positions = {}    # smoothed output from low pass filter

# 5. Filter parameters
ALPHA = 0.2  # smoothing factor: 0 = very smooth, 1 = no smoothing

# 6. Marker IDs used for rectangle
RECT_MARKER_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

print("[INFO] Starting detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

    current_frame_ids = set()

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rvecs[i][0], tvecs[i][0]  # (3,) each
            tvec_np = np.array(tvec)

            # Low pass filter
            if marker_id in filtered_positions:
                filtered_positions[marker_id] = (1 - ALPHA) * filtered_positions[marker_id] + ALPHA * tvec_np
            else:
                filtered_positions[marker_id] = tvec_np.copy()

            marker_positions[marker_id] = filtered_positions[marker_id]
            current_frame_ids.add(marker_id)

            # Draw marker and axis for visualization
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    if ids is not None and len(current_frame_ids.intersection(RECT_MARKER_IDS)) >= 2:
        # Project visible marker points to image
        projected_pts = {}

        for marker_id in RECT_MARKER_IDS:
            if marker_id in current_frame_ids and marker_id in filtered_positions:
                pt3d = filtered_positions[marker_id].reshape(1, 3)
                rvec_zero = np.zeros((3, 1), dtype=np.float32)
                tvec_zero = np.zeros((3, 1), dtype=np.float32)
                pt2d, _ = cv2.projectPoints(pt3d, rvec_zero, tvec_zero, camera_matrix, dist_coeffs)
                projected_pts[marker_id] = tuple(pt2d[0][0].astype(int))

        # Draw lines between visible consecutive markers
        for i in range(len(RECT_MARKER_IDS)):
            id1 = RECT_MARKER_IDS[i]
            id2 = RECT_MARKER_IDS[(i + 1) % len(RECT_MARKER_IDS)]
            if id1 in projected_pts and id2 in projected_pts:
                cv2.line(frame, projected_pts[id1], projected_pts[id2], (0, 255, 0), 2)

    cv2.imshow("Aruco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
