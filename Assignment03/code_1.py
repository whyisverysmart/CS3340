import cv2
import numpy as np
# from sklearn.cluster import KMeans

image = cv2.imread("./image/LP1.jpg")
H, W = image.shape[:2]
image_area = H * W

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(hsv, (100, 180, 170), (130, 255, 255))


# h1 = H // 200
# w1 = W // 400
# print(h1, w1)
kernel = np.ones((4, 4), np.uint8)
mask = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2)

# h2 = H // 200
# w2 = W // 200
# print(h2, w2)
kernel = np.ones((6, 12), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

final_mask = np.zeros_like(mask)
max_area = 0
max_box = None
for cnt in contours:
    rect = cv2.minAreaRect(cnt)  # center (x, y), (w, h), theta
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    w, h = rect[1]
    assert w > 0 and h > 0, "Width and height must be positive"
    ratio = max(w, h) / min(w, h)
    area = w * h

    if 2 < ratio < 6:
        if area > max_area:
            max_area = area
            max_box = box

cv2.drawContours(final_mask, [max_box], 0, 255, -1)

final_img = cv2.bitwise_and(image, image, mask=final_mask)

cv2.imwrite("./output_imgs/LP3_mask.jpg", final_mask)
cv2.imwrite("./output_imgs/LP3_final.jpg", final_img)

# h, w, _ = image.shape
# img_flat = image.reshape((-1, 3)).astype(np.float32)

# K = 3
# kmeans = KMeans(n_clusters=K, random_state=42).fit(img_flat)
# labels = kmeans.labels_
# centers = np.uint8(kmeans.cluster_centers_)

# blue_ref = np.array([235, 123, 11])
# distances = np.linalg.norm(centers - blue_ref, axis=1)
# blue_cluster = np.argmin(distances)

# mask = (labels == blue_cluster).astype(np.uint8).reshape((h, w)) * 255

