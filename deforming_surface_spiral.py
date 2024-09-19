from spiral import generate_spiral_matrix
from derivative_filter import derivative_filter, normalize_to_uint8
from gaussian_blur import gaussian_blur


if __name__ == '__main__':
    import cv2
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

    # test
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    step = 10
    linewidth = 30

    spiral_matrix = generate_spiral_matrix(
        width=1024, height=1024, step=step, linewidth=linewidth)

    # Blur the spiral matrix
    spiral_blur = gaussian_blur(spiral_matrix, 41)
    spiral_blur = cv2.cvtColor(spiral_blur, cv2.COLOR_BGR2GRAY)

    # normalizar spiral_blur al rango -1 a 1
    spiral_blur = np.float32(spiral_blur)
    spiral_blur = 2 * (spiral_blur - np.min(spiral_blur)) / \
        (np.max(spiral_blur) - np.min(spiral_blur)) - 1

    # get the derivatives
    spiral_dx, spiral_dy = derivative_filter(spiral_blur)

    # a * S(x,y) * Dx(S(x,y))
    # a * S(x,y) * Dy(S(x,y))
    a = 12
    map_x = spiral_blur * spiral_dx
    map_y = spiral_blur * spiral_dy
    map_x *= a
    map_y *= a

    # TODO:
    # return map_x and map_y

    output = np.zeros(image.shape, dtype=np.uint8)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            value_x = i + round(map_x[i][j])
            value_y = j + round(map_y[i][j])
            output[i][j] = image[value_x][value_y]

    spiral_dx = normalize_to_uint8(spiral_dx)
    spiral_dy = normalize_to_uint8(spiral_dy)
    map_x = normalize_to_uint8(map_x)
    map_y = normalize_to_uint8(map_y)

    # unir todos los resultados en un solo plot con titulo
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(spiral_matrix)
    axs[0, 0].set_title('Spiral')
    axs[0, 1].imshow(spiral_blur, cmap='gray')
    axs[0, 1].set_title('Spiral Blur')
    axs[1, 0].imshow(spiral_dx, cmap='gray')
    axs[1, 0].set_title('Spiral Dx')
    axs[1, 1].imshow(spiral_dy, cmap='gray')
    axs[1, 1].set_title('Spiral Dy')
    axs[2, 0].imshow(map_x, cmap='gray')
    axs[2, 0].set_title(f'a = {a}* Spiral Dx * Spiral Blur')
    axs[2, 1].imshow(map_y, cmap='gray')
    axs[2, 1].set_title(f'a = {a} * Spiral Dy * Spiral Blur')

    plt.show()
    cv2.imshow('Output', output)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # plt.close(fig)
            break
