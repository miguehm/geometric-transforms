import cv2
import numpy as np


def normalize_to_uint8(image):
    # Normalizar la imagen al rango 0-255
    image_min = np.min(image)
    image_max = np.max(image)
    image_normalized = 255 * (image - image_min) / (image_max - image_min)
    # Convertir la imagen a uint8
    image_uint8 = np.uint8(image_normalized)
    return image_uint8


def derivative_filter(image):

    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # image_float = np.float32(image)
    image_float = image

    # Aplicar la convolución directamente con cv.filter2D
    derivation_x = cv2.filter2D(image_float, -1, kernel_x)
    derivation_y = cv2.filter2D(image_float, -1, kernel_y)

    # derivation_x += 127
    # derivation_y += 127

    # derivation_x = normalize_to_uint8(derivation_x)
    # derivation_y = normalize_to_uint8(derivation_y)

    return derivation_x, derivation_y


if __name__ == "__main__":
    image = cv2.imread('processed_image.jpg', cv2.IMREAD_GRAYSCALE)

    derivation_x, derivation_y = derivative_filter(image)

    # derivation_x += 127
    # derivation_y += 127
    #
    # derivation_x = normalize_to_uint8(derivation_x)
    # derivation_y = normalize_to_uint8(derivation_y)

    # mostrar imagen procesada
    cv2.imshow('Filtered Image X', derivation_x)
    cv2.imshow('Filtered Image Y', derivation_y)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
