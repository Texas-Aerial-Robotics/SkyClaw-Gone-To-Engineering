import cv2, numpy as np
import cv2.aruco as aruco

def _aruco_self_test():
    dict4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    test = np.zeros((600,600), np.uint8)
    aruco.generateImageMarker(dict4x4, 7, 600, test, 1)
    # Detect on the synthetic marker
    params = aruco.DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dict4x4, params)
        corners, ids, _ = det.detectMarkers(test)
    else:
        corners, ids, _ = aruco.detectMarkers(test, dict4x4, parameters=params)
    print("SELF-TEST:", "OK ids="+str(ids.flatten().tolist()) if ids is not None else "FAILED")
_aruco_self_test()