import cv2
import numpy as np


def derivative_filter(image, axis: str):

    if axis == 'x':
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    elif axis == 'y':
        kernel_x = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

    # Aplicar la convolución directamente con cv.filter2D
    filtered_image = cv2.filter2D(image, -1, kernel_x)

    return filtered_image


if __name__ == "__main__":
    image = cv2.imread('processed_image.jpg', cv2.IMREAD_GRAYSCALE)

    derivation_x = derivative_filter(image, axis='x')
    derivation_y = derivative_filter(image, axis='y')

    derivation_x += 127
    derivation_y += 127

    # mostrar imagen
    # cv2.imshow('Original Image', image)

    # mostrar imagen procesada
    cv2.imshow('Filtered Image X', derivation_x)
    cv2.imshow('Filtered Image Y', derivation_y)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
