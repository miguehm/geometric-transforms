from spiral import generate_spiral_matrix
from derivative_filter import derivative_filter, normalize_to_uint8
from gaussian_blur import gaussian_blur
import cv2
import multiprocessing


def map_deforming_surface(surface, a, blur_size=41, visualize=False):
    # Blur the spiral matrix
    original_surface = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)

    if blur_size > 1:
        surface = gaussian_blur(
            original_surface, blur_size, single_channel=True)
    # surface_blur = cv2.cvtColor(surface_blur, cv2.COLOR_BGR2GRAY)

    # normalizar surface_blur al rango -1 a 1
    surface = np.float32(surface)
    surface = 2 * (surface - np.min(surface)) / \
        (np.max(surface) - np.min(surface)) - 1

    # get the derivatives
    surface_dx, surface_dy = derivative_filter(surface)

    # a * S(x,y) * Dx(S(x,y))
    # a * S(x,y) * Dy(S(x,y))
    map_x = surface * surface_dx
    map_y = surface * surface_dy
    map_x *= a
    map_y *= a

    if visualize:
        visual_surface_dx = normalize_to_uint8(surface_dx)
        visual_surface_dy = normalize_to_uint8(surface_dy)
        visual_map_x = normalize_to_uint8(map_x)
        visual_map_y = normalize_to_uint8(map_y)

        # unir todos los resultados en un solo plot con titulo
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs[0, 0].imshow(original_surface, cmap='gray')
        axs[0, 0].set_title('Spiral')
        axs[0, 1].imshow(surface, cmap='gray')
        axs[0, 1].set_title('Spiral Blur')
        axs[1, 0].imshow(visual_surface_dx, cmap='gray')
        axs[1, 0].set_title('Spiral Dx')
        axs[1, 1].imshow(visual_surface_dy, cmap='gray')
        axs[1, 1].set_title('Spiral Dy')
        axs[2, 0].imshow(visual_map_x, cmap='gray')
        axs[2, 0].set_title(f'a = {a}* Spiral Dx * Spiral Blur')
        axs[2, 1].imshow(visual_map_y, cmap='gray')
        axs[2, 1].set_title(f'a = {a} * Spiral Dy * Spiral Blur')
        plt.show()

    return map_x, map_y


def deforming_surface_filter(channel, map_x, map_y):
    # test b&w image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros(channel.shape, dtype=np.uint8)
    rows, cols = channel.shape

    # Convertir map_x, map_y a enteros
    value_x = (np.arange(rows)[:, None] + map_x).astype(int)
    value_y = (np.arange(cols) + map_y).astype(int)

    # Asignar los valores directamente en una "operación vectorizada"
    output = channel[value_x, value_y]

    return output


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    image = Image.open('image.jpg')

    # pasar a array de numpy
    image = np.array(image)

    # turn PNG RGBA to BGR
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # turn JPG RGB to BGR
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # create a spiral matrix
    step = 10
    linewidth = 30

    spiral_matrix = generate_spiral_matrix(
        width=1024, height=1024, step=step, linewidth=linewidth)

    map_x, map_y = map_deforming_surface(
        spiral_matrix, blur_size=60, a=40, visualize=True)

    B, G, R = cv2.split(image)

    with multiprocessing.Pool(processes=3) as pool:
        argumentos = [(B, map_x, map_y),
                      (G, map_x, map_y),
                      (R, map_x, map_y)]
        b, g, r = pool.starmap(deforming_surface_filter, argumentos)

    output = cv2.merge((b, g, r))

    cv2.imshow('Output', output)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
