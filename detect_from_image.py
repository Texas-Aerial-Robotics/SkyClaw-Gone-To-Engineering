# detect_from_image.py
import cv2, cv2.aruco as aruco

img = cv2.imread("aruco_4x4_50_id0.png", cv2.IMREAD_GRAYSCALE)
d = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
detector = aruco.ArucoDetector(d, params) if hasattr(aruco, "ArucoDetector") else None

if detector:
    corners, ids, rej = detector.detectMarkers(img)
else:
    corners, ids, rej = aruco.detectMarkers(img, d, parameters=params)

print("ids:", ids)
out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if ids is not None:
    aruco.drawDetectedMarkers(out, corners, ids)
cv2.imshow("file detect", out); cv2.waitKey(0)
