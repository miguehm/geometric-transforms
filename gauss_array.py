import cv2


def gauss_array(elements: int):
    if elements < 1:
        elements = 1
        print(
            f'Elements must be greater than 0. Changed to {elements}')

    # ya esta normalizado
    arr = cv2.getGaussianKernel(elements, -1)

    return arr


if __name__ == '__main__':
    elements = input('Enter the number of elements: ')
    elements = int(elements)
    arr = gauss_array(elements)
    print(f'Array: {arr}')
