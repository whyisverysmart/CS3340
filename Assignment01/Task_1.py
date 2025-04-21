import cv2
import matplotlib.pyplot as plt

image_path = "./image/star.JPG"
image = cv2.imread(image_path)

# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 均衡化
equalized_image = cv2.equalizeHist(gray)
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.plot(hist_equalized, color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 255])
plt.grid()

output_path = "./output_image/1.2histogram_equalized.png"
plt.savefig(output_path)

equalized_image_path = "./output_image/1.2equalized_image.jpg"
cv2.imwrite(equalized_image_path, equalized_image)