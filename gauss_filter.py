import cv2 as cv
import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def gauss_array(m: int) -> tuple:
    try:
        center = m // 2
        sigma = (m - 1) / 6.0
        x = np.arange(-center, center + 1)
        gauss_filter = np.exp(-0.5 * (x / sigma) ** 2)
        # Normalizar para que la suma sea 1
        gauss_filter /= np.sum(gauss_filter)

        return gauss_filter
    except ValueError as e:
        print(f'{e}')
        return ([], None)
    except TypeError as e:
        print(f'{e}')
        return ([], None)


def to_gauss_filter(img, gauss_size: int):
    # Convertir la imagen PIL a una matriz NumPy
    opencv_image = np.array(img)

    # Verificar el número de canales en la imagen
    if opencv_image.ndim == 3 and opencv_image.shape[2] == 4:
        # Convertir de BGRA a BGR (ignorando el canal alfa)
        opencv_image = cv.cvtColor(opencv_image, cv.COLOR_RGBA2BGR)
    elif opencv_image.ndim == 3 and opencv_image.shape[2] == 3:
        # Convertir de RGB a BGR si es necesario
        opencv_image = cv.cvtColor(opencv_image, cv.COLOR_RGB2BGR)
    elif opencv_image.ndim == 2:
        # La imagen es en escala de grises
        return img
    else:
        raise ValueError("Formato de imagen no soportado.")

    # Separar los canales B, G, R
    b, g, r = cv.split(opencv_image)

    # gauss_size = 3

    if gauss_size % 2 == 0:
        raise ValueError("El tamaño del filtro Gaussiano debe ser impar.")

    # Obtener el filtro Gaussiano unidimensional
    gauss_filter = gauss_array(gauss_size)

    # Convertir el filtro 1D a un filtro 2D
    gauss_filter_2d = np.outer(gauss_filter, gauss_filter)

    # Normalizar el filtro Gaussiano 2D
    gauss_filter_2d = gauss_filter_2d / np.sum(gauss_filter_2d)

    # Aplicar la convolución a cada canal usando el filtro Gaussiano 2D
    output_b = convolve(b, gauss_filter_2d, mode='constant', cval=0.0)
    output_g = convolve(g, gauss_filter_2d, mode='constant', cval=0.0)
    output_r = convolve(r, gauss_filter_2d, mode='constant', cval=0.0)

    # Recombinar los canales en una imagen
    output = cv.merge((output_b, output_g, output_r))

    # Convertir la imagen de BGR a RGB
    output = cv.cvtColor(output, cv.COLOR_BGR2RGB)

    # Asegurarse de que los valores estén en el rango adecuado
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Convertir la matriz NumPy a imagen PIL
    output_image = Image.fromarray(output)

    return output_image
