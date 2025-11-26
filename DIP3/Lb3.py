import cv2 as cv
import matplotlib.pyplot as plt


def quick_rotate_filter(image_path, angle):

    img = cv.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить {image_path}")
        return

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))

    filters = {
        'Повернутое': rotated,
        'Гаусс 5x5': cv.GaussianBlur(rotated, (5, 5), 0),
        'Гаусс 9x9': cv.GaussianBlur(rotated, (9, 9), 0),
        'Медиана 5': cv.medianBlur(rotated, 5),
        'Усреднение 7x7': cv.blur(rotated, (7, 7)),
    }

    plt.figure(figsize=(15, 8))
    for i, (name, result) in enumerate(filters.items(), 1):
        plt.subplot(2, 3, i)
        display = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        plt.imshow(display)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "original.jpg"
    angle = 260
    quick_rotate_filter(image_path, angle)