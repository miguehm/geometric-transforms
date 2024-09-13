import numpy as np


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
