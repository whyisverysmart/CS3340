import cv2
import numpy as np

image = cv2.imread("./image/plate.jpg")
resized = cv2.resize(image, (400, 300))

hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
# mask_blue = cv2.inRange(hsv, (100, 180, 170), (130, 255, 255))
mask_blue = cv2.inRange(hsv, (100, 180, 200), (130, 255, 255))
# mask_white = cv2.inRange(hsv, (0, 0, 220), (180, 30, 255))
# mask = cv2.bitwise_and(mask_white, mask_blue)

kernel = np.ones((3, 1), np.uint8)
mask = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2)
kernel = np.ones((3, 6), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)
# cv2.imwrite("./output_image/2.2_plate_mask.jpg", mask)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
edges_canny = cv2.Canny(gray, 100, 250, L2gradient=True)
edges_canny = cv2.bitwise_and(edges_canny, mask)
cv2.imwrite("./output_image/2.2_plate_canny_edges.jpg", edges_canny)


# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# sobel_combined = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)
# cv2.imwrite("./output_image/2.2_plate_sobel_edges.jpg", sobel_combined)

# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# laplacian_abs = cv2.convertScaleAbs(laplacian)
# cv2.imwrite("./output_image/2.2_plate_laplacian_edges.jpg", laplacian_abs)