import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tangerine.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Перевод в LAB
lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(lab)

mask = np.zeros_like(L, dtype=np.uint8)
mask[(L > 120) & (A > 140) & (B > 160)] = 255

mask_expanded = cv2.dilate(mask, np.ones((25, 25), np.uint8)) # эти три строчки к маске мандаринов (ТОЛЬКО мандаринов)
bright_pixels = (L > 200)                                            # добавляют пиксели светлее мандаринов
mask[(bright_pixels) & (mask_expanded > 0)] = 255                    # чтоб дырок не было и бликов кувшина не добавились

kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

result = img_rgb.copy()
result[mask == 0] = (0, 0, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("Исходное")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Маска мандаринов")
axes[1].axis("off")

axes[2].imshow(result)
axes[2].set_title("Мандарины (фон черный)")
axes[2].axis("off")

plt.savefig("result.png")
plt.show()