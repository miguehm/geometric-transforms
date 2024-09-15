import multiprocessing
import cv2
from gauss_array import gauss_array
from gauss_filter import gauss_filter


def gaussian_blur(image, elements: int):
    ARR = gauss_array(elements)

    # split image
    B, G, R = cv2.split(image)

    # array size
    rows, cols = B.shape

    print('Start processing...')
    with multiprocessing.Pool(processes=3) as pool:
        argumentos = [(B, ARR),
                      (G, ARR),
                      (R, ARR)]
        b, g, r = pool.starmap(gauss_filter, argumentos)

    print('End processing...')

    image = cv2.merge((b, g, r))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    image = Image.open('image.jpg')

    # pasar a array de numpy
    image = np.array(image)

    # turn PNG RGBA to BGR
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # turn JPG RGB to BGR
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = gaussian_blur(image, int(input('Enter the number of elements: ')))
    cv2.imwrite('processed_image.jpg', result)
    print('Image saved as processed_image.jpg')
