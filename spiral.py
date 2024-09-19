import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2


def generate_spiral_matrix(width: int, height: int, step: int, linewidth: int):
    # Definir los valores de alpha (ángulo)
    alpha = np.linspace(0, step * np.pi, 1000)

    # Definir el radio en función de alpha (por ejemplo, r = a * alpha)
    a = 0.1
    r = a * alpha

    # Calcular las coordenadas x e y en el plano cartesiano
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)

    # Definir la resolución (dpi)
    dpi = 100

    # Crear la figura con el tamaño en pulgadas
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Plotear la espiral
    ax.plot(x, y, color='black', linewidth=linewidth)

    # Ocultar los ejes y el título
    ax.axis('off')

    # Ajustar los márgenes de la figura
    # para asegurar que se guarde todo el contenido
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Renderizar la figura y convertirla a una matriz
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(),
                          dtype=np.uint8).reshape(height, width, 4)

    # delete canvas
    plt.close(fig)

    # Convertir la imagen de RGBA a BGR (formato usado por OpenCV)
    # ignorando el canal alfa
    image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGBA2BGR)

    return image_bgr


if __name__ == '__main__':
    # Obtener la matriz de la espiral
    spiral_matrix = generate_spiral_matrix(
        width=1200, height=1200, step=12, linewidth=10)

    # Mostrar la matriz usando OpenCV (opcional)
    cv2.imshow('Spiral', spiral_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
