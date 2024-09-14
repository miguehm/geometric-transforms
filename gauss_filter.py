import cv2 as cv
import numpy as np


def gauss_filter(image, DATA):
    # Crear kernel 2D a partir de la 1D
    kernel = np.outer(DATA['array'], DATA['array'])

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


# def gauss_filter(image, DATA, rows, cols):
#     # set padding
#     padded_img = cv.copyMakeBorder(
#         image,
#         DATA['top_padding'],
#         DATA['bottom_padding'],
#         DATA['left_padding'],
#         DATA['right_padding'],
#         cv.BORDER_CONSTANT,
#         value=0
#     )
#
#     # print(f'Padded image:\n{padded_img}')
#
#     for i in range(rows):
#         for j in range(cols):
#             row = padded_img[DATA['top_padding'] + i]
#
#             # get region of interest
#             roi = row[DATA['left_padding'] + j - DATA['left_padding']:
#                       DATA['left_padding'] + j + DATA['right_padding'] + 1]
#
#             convolution = round(np.sum(roi * DATA['array']))
#             image[i][j] = convolution
#
#     # print(f'Filtered image gauss row:\n{image}')
#
#     image = image.T
#
#     # print(f'Filtered image transposed:\n{image}')
#
#     padded_img = cv.copyMakeBorder(
#         image,
#         DATA['top_padding'],
#         DATA['bottom_padding'],
#         DATA['left_padding'],
#         DATA['right_padding'],
#         cv.BORDER_CONSTANT,
#         value=0
#     )
#
#     rows, cols = cols, rows
#
#     for i in range(rows):
#         for j in range(cols):
#             row = padded_img[DATA['top_padding'] + i]
#
#             # get region of interest
#             roi = row[DATA['left_padding'] + j - DATA['left_padding']:
#                       DATA['left_padding'] + j + DATA['right_padding'] + 1]
#             # print(f'ROI: {roi}')
#
#             convolution = round(np.sum(roi * DATA['array']))
#
#             # print(f'Convolution: {convolution}')
#
#             image[i][j] = convolution
#
#     image = image.T
#
#     # print(f'Filtered image transposed back gauss column:\n{image}')
#
#     return image
#
#
# if __name__ == "__main__":
#     from PIL import Image
#     from gauss_array import gauss_array
#
#     image = Image.open('image.jpg')
#
#     # pasar a array de numpy
#     image = np.array(image)
#
#     image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#
#     elements = int(input('Enter the number of elements: '))
#     DATA = gauss_array(elements)
#
#     image = gauss_filter(image, DATA, image.shape[0], image.shape[1])
#
#     # Save the processed image as file
#     cv.imwrite('processed_image.jpg', image)
