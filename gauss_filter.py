import cv2 as cv
import numpy as np


def gauss_filter(image, ARR):
    # Crear kernel 2D a partir de la 1D
    kernel = np.outer(ARR, ARR)

    # Aplicar la convolución directamente con cv.filter2D
    filtered_image = cv.filter2D(image, -1, kernel)

    return filtered_image


if __name__ == "__main__":
    from PIL import Image
    from gauss_array import gauss_array

    image = Image.open('image.jpg')

    # Convertir a array de numpy y escala de grises
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    elements = int(input('Enter the number of elements: '))
    DATA = gauss_array(elements)

    image = gauss_filter(image, DATA)

    # Guardar la imagen procesada
    cv.imwrite('processed_image.jpg', image)
