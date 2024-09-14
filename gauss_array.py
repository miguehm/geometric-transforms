import cv2


def gauss_array(elements: int):
    if elements < 1:
        elements = 1
        print(
            f'Elements must be greater than 0. Changed to {elements}')

    arr = cv2.getGaussianKernel(elements, -1)

    # pasar vector vector columna a vector fila
    ARR = arr.T[0]

    ANCHOR_POS = (elements - 1) // 2
    ARR_BEFORE_ANCHOR = ARR[:ANCHOR_POS]
    ARR_AFTER_ANCHOR = ARR[ANCHOR_POS + 1:]
    LEFT_PADDING = len(ARR_BEFORE_ANCHOR)
    RIGHT_PADDING = len(ARR_AFTER_ANCHOR)
    TOP_PADDING = LEFT_PADDING
    BOTTOM_PADDING = RIGHT_PADDING

    data = {
        'elements': elements,
        'array': ARR,
        'anchor_pos': ANCHOR_POS,
        'arr_before_anchor': ARR_BEFORE_ANCHOR,
        'arr_after_anchor': ARR_AFTER_ANCHOR,
        'left_padding': LEFT_PADDING,
        'right_padding': RIGHT_PADDING,
        'top_padding': TOP_PADDING,
        'bottom_padding': BOTTOM_PADDING
    }

    # ya esta normalizado
    return data

    # if elements > 63:
    #     elements = 63
    #     print(
    #         f'Number of elements must be less than 64. Changed to {elements}')

    # if elements % 2 == 0:
    #     elements -= 1
    #     print(f'Number of elements must be even. Changed to {elements}')

    # n = elements - 1
    # nfactorial = factorial(n)
    #
    # arr = np.zeros(elements).astype(int)
    # for i in range(elements):
    #     arr[i] = nfactorial / (factorial(i) * factorial(n - i))
    #
    # return {
    #     'array': arr,
    #     'total': int(np.sum(arr))
    # }


if __name__ == '__main__':
    elements = input('Enter the number of elements: ')
    elements = int(elements)
    arr = gauss_array(elements)
    print(f'Array: {arr}')
