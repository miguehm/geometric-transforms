import cv2
import numpy as np


def crystal_transform(image, block_size=60):
    height, width, _ = image.shape

    # Crear las coordenadas x, y
    x = np.arange(width)
    y = np.arange(height)

    # Crear una malla con las coordenadas
    x_grid, y_grid = np.meshgrid(x, y)

    # Calcular las nuevas coordenadas con operaciones vectorizadas
    new_x = x_grid - (x_grid % block_size) + (y_grid % block_size)
    new_y = y_grid - (y_grid % block_size) + (x_grid % block_size)

    # Asegurarse de que las nuevas coordenadas estén dentro de los límites
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    # Aplicar la transformación
    transformed_image = np.zeros_like(image)
    transformed_image[y_grid, x_grid] = image[new_y, new_x]

    return transformed_image


if __name__ == '__main__':
    # Cargar la imagen
    image = cv2.imread('./image.jpg')

    # Aplicar la transformación
    transformed_image = crystal_transform(image, block_size=60)

    # Mostrar la imagen original y la transformada
    cv2.imshow('Original Image', image)
    cv2.imshow('Transformed Image', transformed_image)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
