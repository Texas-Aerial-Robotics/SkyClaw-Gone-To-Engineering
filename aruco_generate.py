import cv2, numpy as np
aruco = cv2.aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 23           # pick any id (0–249)
size_px = 1024           # resolution
img = aruco.generateImageMarker(dict, marker_id, size_px)
cv2.imwrite("/home/tarlaptop/TAR_team1/SkyClaw-Gone-To-Engineering/marker_23.png", img)
print("Wrote marker_23.png")
