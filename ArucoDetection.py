#----

# Run on laptop webcam with ArUco detection + pose axes (OpenCV 4.x safe)
import cv2
import cv2.aruco as aruco
import numpy as np


# ---------------- Parameters you should set ----------------
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
video = ""        # keep empty for camera
camId = 0         # 0 is usually the built-in webcam
markerLength = 0.05  # meters (CHANGE to your print size)
estimatePose = False
showRejected = True

# 🔧 Replace with your real calibration for accurate pose
# Example from calibration: [[fx,0,cx],[0,fy,cy],[0,0,1]]
data = np.load("webcam_chessboard_calib_1280x720.npz")  # <-- filename must match yours
camMatrix  = data["K"]
distCoeffs = data["dist"]

# ---------------- Open camera robustly ----------------
# Try a good backend per OS; fall back to default if needed.
# Windows: CAP_DSHOW, macOS: CAP_AVFOUNDATION, Linux: CAP_V4L2
cap_flags = [
    cv2.CAP_DSHOW,          # Windows
    cv2.CAP_AVFOUNDATION,   # macOS
    cv2.CAP_V4L2            # Linux
]

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Left camera image
        img_ocv = image.get_data()[:, :, :3]     # Convert to OpenCV format
        cv2.imshow("ZED Image", img_ocv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
#if video:
#    inputVideo = cv2.VideoCapture(video)
#else:
    # Try backends until one opens
    #inputVideo = None
    #for flag in cap_flags:
       # cap = cv2.VideoCapture(camId, flag)
       # if cap.isOpened():
          #  inputVideo = cap
         #   break
    #if inputVideo is None:
        # last resort: default
      #  inputVideo = cv2.VideoCapture(camId)

#if not inputVideo.isOpened():
   # raise RuntimeError("Could not open webcam. Try a different camId (1,2,...) or check permissions.")


# Optional: set a sane resolution (comment out if you prefer default)
#inputVideo.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#inputVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

waitTime = 0 if video else 1
totalTime = 0.0
totalIterations = 0

# ---------------- Handle both detector APIs ----------------
use_new_api = hasattr(aruco, "ArucoDetector")  # OpenCV ≥ 4.7
if use_new_api:
    detector = aruco.ArucoDetector(dictionary, detectorParams)

#while inputVideo.isOpened():
#    ret, frame = inputVideo.read()
#    if not ret:
#        break

    image = img_ocv  # BGR is fine; ArUco works on color
    imageCopy = image.copy()

    tick = cv2.getTickCount()
    # Detect markers
    if use_new_api:
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = aruco.detectMarkers(image, dictionary, parameters=detectorParams)
    if ids is None or len(ids) == 0:
        print("Markers: 0")
    else:
        print(f"Markers: {len(ids)} IDs: {ids.flatten().tolist()}")    
    
        

    # Pose (if requested and we saw something)
    if estimatePose and ids is not None and len(ids) > 0:
        # Simpler & robust: estimate per marker directly
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs)
    else:
        rvecs, tvecs = None, None

    # Timing log
    currentTime = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
    totalTime += currentTime
    totalIterations += 1
    if totalIterations % 30 == 0:
        print(f"Detection Time = {currentTime * 1000:.2f} ms "
              f"(Mean = {1000 * totalTime / totalIterations:.2f} ms)")

    # Draw results
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(imageCopy, corners, ids)
        if estimatePose and rvecs is not None:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5, 2)

    if showRejected and rejected is not None and len(rejected) > 0:
        aruco.drawDetectedMarkers(imageCopy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("ArUco Webcam", imageCopy)
    cv2.namedWindow("ArUco Webcam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ArUco Webcam", cv2.WND_PROP_TOPMOST, 1)
    key = cv2.waitKey(waitTime) & 0xFF
    if key == 27:  # ESC to quit
        break

    # After: corners, ids, rejected = ...
count = 0 if ids is None else len(ids)
if count == 0:
    print("Markers: 0")
else:
    print(f"Markers: {count} IDs: {ids.flatten().tolist()}")

# Draw & save for proof
if ids is not None and len(ids) > 0:
    aruco.drawDetectedMarkers(imageCopy, corners, ids)
    if estimatePose:
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5, 2)
    cv2.imwrite("debug_detected_frame.png", imageCopy)  # check file in your script folder

# Make window easy to notice
cv2.namedWindow("ArUco Webcam", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ArUco Webcam", cv2.WND_PROP_TOPMOST, 1)
cv2.imshow("ArUco Webcam", imageCopy)

inputVideo.release()
cv2.destroyAllWindows()
