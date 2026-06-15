import cv2
import numpy as np

# Load image
img = cv2.imread('webui/storage/batches/14c2665cdad3/inputs/image_8_.png')
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Try ArUco detection on original
import cv2.aruco as aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary)

corners_orig, ids_orig, _ = detector.detectMarkers(gray)
print("Original image - Markers:", ids_orig.flatten().tolist() if ids_orig is not None else "NONE")

# Try on rotated image
img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
gray_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
corners_rot, ids_rot, _ = detector.detectMarkers(gray_rot)
print("Rotated 90 CW - Markers:", ids_rot.flatten().tolist() if ids_rot is not None else "NONE")
