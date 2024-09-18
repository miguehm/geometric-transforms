import numpy as np
import matplotlib.pyplot as plt

# Definir los valores de alpha (ángulo)
alpha = np.linspace(0, 10 * np.pi, 1000)

# Definir el radio en función de alpha (por ejemplo, r = a * alpha)
a = 0.1
r = a * alpha

# Calcular las coordenadas x e y en el plano cartesiano
x = r * np.cos(alpha)
y = r * np.sin(alpha)

# Definir el tamaño de la imagen en píxeles
width, height = 800, 800

# Definir la resolución (dpi)
dpi = 100

# Crear la figura con el tamaño en pulgadas
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

# Plotear la espiral
plt.plot(x, y, color='black', linewidth=25)

# Ocultar los ejes y el título
plt.axis('off')

# Ajustar los márgenes de la figura para asegurar que se guarde todo el contenido
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Guardar la gráfica como una imagen JPG con la resolución especificada
plt.savefig('spiral.jpg', format='jpg', dpi=dpi)

# Mostrar la gráfica (opcional)
# plt.show()
