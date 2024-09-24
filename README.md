# Transformaciones Geométricas de Imágenes con OpenCV

## Introducción

Las transformaciones geométricas son técnicas fundamentales en el procesamiento digital de imágenes que permiten modificar la apariencia y estructura de una imagen. OpenCV, una biblioteca de visión por computadora de código abierto, ofrece una amplia gama de herramientas para aplicar estas transformaciones de manera eficiente. En este documento, nos centraremos en tres transformaciones específicas: el suavizado gaussiano, la superficie deformante y el efecto cristal.

Estas técnicas tienen aplicaciones diversas, desde la mejora de la calidad de imagen hasta la creación de efectos visuales especiales. El suavizado gaussiano se utiliza comúnmente para reducir el ruido y suavizar los detalles de una imagen. La superficie deformante permite distorsionar la imagen de manera controlada, creando efectos de ondulación o deformación. El efecto cristal, por su parte, simula la apariencia de una imagen vista a través de un cristal texturizado.

## Resumen

Este estudio examina tres transformaciones geométricas aplicadas a imágenes utilizando la biblioteca OpenCV:

1. **Suavizado Gaussiano**: Se analiza la aplicación de un filtro gaussiano para reducir el ruido y suavizar los detalles de la imagen. Se discuten los parámetros clave como el tamaño del kernel y la desviación estándar, y su impacto en el resultado final.

2. **Superficie Deformante**: Se explora la técnica de deformación de imágenes mediante la aplicación de una función de mapeo no lineal. Se presentan diferentes tipos de deformaciones y se estudia cómo afectan la geometría original de la imagen.

3. **Efecto Cristal**: Se investiga la simulación de un efecto de cristal texturizado sobre la imagen. Se examinan los métodos para generar patrones de desplazamiento aleatorios y cómo estos se aplican para crear la ilusión de una superficie de cristal.

Para cada transformación, se proporcionan ejemplos de implementación utilizando las funciones de OpenCV, se discuten los desafíos técnicos y se evalúan los resultados visuales. Además, se consideran las posibles aplicaciones prácticas de estas técnicas en campos como el procesamiento de imágenes médicas, la fotografía digital y los efectos visuales en la industria del entretenimiento.

# Diseño

## Suavisado gaussiano

El suavisado gaussiano requiere la generación de un kernel gaussiano 1D:

```python
def gauss_array(elements: int):
    if elements < 1:
        elements = 1
        print(
            f'Elements must be greater than 0. Changed to {elements}')

    # Ya está normalizado
    arr = cv2.getGaussianKernel(elements, -1)

    return arr
```

La función `gauss_array()` definida en `gauss_array.py` toma un número de elementos que determina el tamaño del kernel gaussiano. Utiliza la función `cv2.getGaussianKernel()` de OpenCV para generar el array 1D gaussiano, que luego se devuelve.

En la función `gaussian_blur()`, se crea el kernel 2D a partir del array 1D usando `np.outer()`:

```python
def gaussian_blur(image, elements: int, single_channel=False):
    ARR = gauss_array(elements)

    if single_channel:
        return gauss_filter(image, ARR)

    # Dividir la imagen en canales
    B, G, R = cv2.split(image)

    # Aplicar el filtro gaussiano a cada canal por separado
    with multiprocessing.Pool(processes=3) as pool:
        argumentos = [(B, ARR),
                      (G, ARR),
                      (R, ARR)]
        b, g, r = pool.starmap(gauss_filter, argumentos)

    # Combinar los canales filtrados
    image = cv2.merge((b, g, r))

    return image
```

Si la imagen tiene múltiples canales (RGB), la función `gaussian_blur()` divide la imagen en canales individuales (B, G, R) utilizando `cv2.split()`. Luego, aplica el filtrado gaussiano a cada canal por separado usando la función auxiliar `gauss_filter()` definida en `gauss_filter.py`:

```python
def gauss_filter(image, ARR):
    # Crear kernel 2D a partir de la 1D
    kernel = np.outer(ARR, ARR)

    # Aplicar la convolución directamente con cv.filter2D
    filtered_image = cv.filter2D(image, -1, kernel)

    return filtered_image
```

La función `gauss_filter()` toma la imagen y el array 1D gaussiano, crea el kernel 2D mediante `np.outer()` y aplica la convolución utilizando `cv2.filter2D()`. Esto suaviza la imagen aplicando el filtro gaussiano.

Finalmente, la función `gaussian_blur()` combina los canales filtrados utilizando `cv2.merge()` y devuelve la imagen procesada.

Para mejorar el rendimiento, la implementación utiliza `multiprocessing.Pool` para procesar los canales de manera paralela. Esto aprovecha los múltiples núcleos de la CPU y reduce significativamente el tiempo de procesamiento, especialmente en imágenes de gran tamaño.

## Superficie Deformante

La transformación de superficie deformante es una técnica que permite distorsionar la apariencia de una imagen de una manera controlada, creando efectos de ondulación o deformación.

La función principal es `deforming_surface_spiral()`, que toma una imagen de entrada y aplica la deformación de superficie. Veamos cómo funciona:

1. Primero, la función genera una matriz en espiral utilizando la función `generate_spiral_matrix()` definida en el archivo "spiral.py":

   ```python
   def deforming_surface_spiral(image, step=10, linewidth=30, a=41):
       # create a spiral matrix
       spiral_matrix = generate_spiral_matrix(
           width=width, height=height, step=step, linewidth=linewidth)
   ```

   Esta matriz en espiral servirá como mapa de desplazamiento para la deformación de la imagen.

El archivo "derivative_filter.py" contiene la función `derivative_filter()` que se encarga de calcular los gradientes (derivadas) de una imagen.

```python
def derivative_filter(image):
    # Definir los kernels de convolución para calcular las derivadas
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Convertir la imagen a punto flotante
    image_float = np.float32(image)

    # Aplicar la convolución con los kernels para calcular las derivadas
    derivation_x = cv2.filter2D(image_float, -1, kernel_x)
    derivation_y = cv2.filter2D(image_float, -1, kernel_y)

    return derivation_x, derivation_y
```

La función `derivative_filter()` utiliza los kernels de Sobel definidos en `kernel_x` y `kernel_y` para calcular las derivadas de la imagen a lo largo de los ejes X e Y, respectivamente. Esto se logra aplicando la convolución de la imagen con estos kernels utilizando la función `cv2.filter2D()` de OpenCV.

El resultado son dos imágenes que representan las derivadas (gradientes) de la imagen de entrada a lo largo de los ejes X e Y. Estas derivadas se utilizan posteriormente en la transformación de superficie deformante para calcular los mapas de desplazamiento.

El archivo "spiral.py" contiene la función `generate_spiral_matrix()` que se encarga de generar una matriz en espiral que se utilizará como mapa de desplazamiento para la transformación de superficie deformante.

```python
def generate_spiral_matrix(width: int, height: int, step: int, linewidth: int):
    # Definir los valores de alpha (ángulo)
    alpha = np.linspace(0, step * np.pi, 1000)

    # Definir el radio en función de alpha (por ejemplo, r = a * alpha)
    a = 0.1
    r = a * alpha

    # Calcular las coordenadas x e y en el plano cartesiano
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)

    # Crear la figura y renderizarla a una matriz de imagen
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='black', linewidth=linewidth)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)

    # Convertir la imagen de RGBA a BGR (formato usado por OpenCV)
    image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGBA2BGR)

    return image_bgr
```

La función `generate_spiral_matrix()` sigue estos pasos:

1. Calcula los valores de ángulo `alpha` en un rango determinado por el parámetro `step`.
2. Calcula las coordenadas `x` e `y` de una espiral utilizando las fórmulas `x = r * cos(alpha)` y `y = r * sin(alpha)`, donde `r` es una función del ángulo `alpha`.
3. Crea una figura de Matplotlib y la renderiza a una matriz de imagen.
4. Convierte la imagen de RGBA a BGR (formato utilizado por OpenCV) y la devuelve.

El resultado es una matriz de imagen que representa una espiral, que luego se utiliza como mapa de desplazamiento en la transformación de superficie deformante.

A continuación, la función `map_deforming_surface()` calcula los mapas de desplazamiento x e y a partir de la matriz en espiral:

```python
def map_deforming_surface(surface, a, blur_size=41, visualize=False):
    # Normalizar la matriz en espiral al rango -1 a 1
    surface = np.float32(surface)
    surface = 2 * (surface - np.min(surface)) / (np.max(surface) - np.min(surface)) - 1

    # Calcular los gradientes (derivadas) de la matriz en espiral
    surface_dx, surface_dy = derivative_filter(surface)

    # Aplicar la fórmula de deformación
    map_x = surface * surface_dx
    map_y = surface * surface_dy
    map_x *= a
    map_y *= a
```

La función utiliza la función `derivative_filter()` definida en `derivative_filter.py` para calcular los gradientes (derivadas) de la matriz en espiral. Luego, aplica la fórmula de deformación utilizando estos gradientes y un factor de escala `a`.

El archivo "derivative_filter.py" contiene la función `derivative_filter()` que se encarga de calcular los gradientes (derivadas) de una imagen.

```python
def derivative_filter(image):
    # Definir los kernels de convolución para calcular las derivadas
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Convertir la imagen a punto flotante
    image_float = np.float32(image)

    # Aplicar la convolución con los kernels para calcular las derivadas
    derivation_x = cv2.filter2D(image_float, -1, kernel_x)
    derivation_y = cv2.filter2D(image_float, -1, kernel_y)

    return derivation_x, derivation_y
```

La función `derivative_filter()` utiliza los kernels de Sobel definidos en `kernel_x` y `kernel_y` para calcular las derivadas de la imagen a lo largo de los ejes X e Y, respectivamente. Esto se logra aplicando la convolución de la imagen con estos kernels utilizando la función `cv2.filter2D()` de OpenCV.

El resultado son dos imágenes que representan las derivadas (gradientes) de la imagen de entrada a lo largo de los ejes X e Y. Estas derivadas se utilizan posteriormente en la transformación de superficie deformante para calcular los mapas de desplazamiento.

3. Finalmente, la función `deforming_surface_filter()` se encarga de aplicar la deformación a cada canal de la imagen de entrada:

   ```python
   def deforming_surface_filter(channel, map_x, map_y):
       # Aplicar la deformación a cada canal de la imagen
       output = np.zeros(channel.shape, dtype=np.uint8)
       rows, cols = channel.shape

       # Calcular los nuevos índices de los píxeles utilizando los mapas de desplazamiento
       value_x = (np.arange(rows)[:, None] + map_x).astype(int)
       value_y = (np.arange(cols) + map_y).astype(int)

       # Asignar los valores de los píxeles en la imagen deformada
       output = channel[value_x, value_y]

       return output
   ```

   Esta función utiliza una operación vectorizada para reasignar los valores de los píxeles en función de los mapas de desplazamiento x e y calculados anteriormente.

El archivo "spiral.py" contiene la función `generate_spiral_matrix()` que se encarga de generar una matriz en espiral que se utilizará como mapa de desplazamiento para la transformación de superficie deformante.

```python
def generate_spiral_matrix(width: int, height: int, step: int, linewidth: int):
    # Definir los valores de alpha (ángulo)
    alpha = np.linspace(0, step * np.pi, 1000)

    # Definir el radio en función de alpha (por ejemplo, r = a * alpha)
    a = 0.1
    r = a * alpha

    # Calcular las coordenadas x e y en el plano cartesiano
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)

    # Crear la figura y renderizarla a una matriz de imagen
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='black', linewidth=linewidth)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)

    # Convertir la imagen de RGBA a BGR (formato usado por OpenCV)
    image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGBA2BGR)

    return image_bgr
```

La función `generate_spiral_matrix()` sigue estos pasos:

1. Calcula los valores de ángulo `alpha` en un rango determinado por el parámetro `step`.
2. Calcula las coordenadas `x` e `y` de una espiral utilizando las fórmulas `x = r * cos(alpha)` y `y = r * sin(alpha)`, donde `r` es una función del ángulo `alpha`.
3. Crea una figura de Matplotlib y la renderiza a una matriz de imagen.
4. Convierte la imagen de RGBA a BGR (formato utilizado por OpenCV) y la devuelve.

El resultado es una matriz de imagen que representa una espiral, que se utiliza como mapa de desplazamiento en la transformación de superficie deformante.

## Efecto Cristal

La transformación de efecto cristal es una técnica que simula la apariencia de una imagen vista a través de una superficie de cristal texturizada. El código para implementar esta transformación se encuentra en el archivo "cristal.py".

La función principal es `crystal_transform()`, que toma una imagen de entrada y aplica el efecto cristal. Veamos cómo funciona:

Primero, la función crea dos matrices de coordenadas `x_grid` y `y_grid` que representan las coordenadas de cada píxel en la imagen:

```python
x = np.arange(width)
y = np.arange(height)
x_grid, y_grid = np.meshgrid(x, y)
```

Luego, se calculan las nuevas coordenadas de los píxeles utilizando una operación vectorizada:

```python
new_x = x_grid - (x_grid % block_size) + (y_grid % block_size)
new_y = y_grid - (y_grid % block_size) + (x_grid % block_size)
```

Esta fórmula crea un patrón de desplazamiento de los píxeles en función del parámetro `block_size`. Cada bloque de píxeles de tamaño `block_size` x `block_size` se desplaza de forma diagonal, creando el efecto de cristal.

Para asegurarse de que las nuevas coordenadas estén dentro de los límites de la imagen, se utiliza la función `np.clip()`:

```python
new_x = np.clip(new_x, 0, width - 1)
new_y = np.clip(new_y, 0, height - 1)
```

Finalmente, se crea una imagen transformada vacía y se asignan los valores de los píxeles de la imagen original a las nuevas coordenadas:

```python
transformed_image = np.zeros_like(image)
transformed_image[y_grid, x_grid] = image[new_y, new_x]
```

Este paso realiza la transformación final, asignando los valores de los píxeles a sus nuevas posiciones en la imagen transformada.

El resultado de esta transformación es una imagen que parece distorsionada por una superficie de cristal texturizada, creando un efecto visual interesante.

El parámetro `block_size` controla el tamaño de los bloques de píxeles que se desplazan, lo que permite ajustar la apariencia y la intensidad del efecto cristal.

# Evaluación

## Suavisado Gaussiano

![[image.jpg]]

> Imagen original

![[Pasted image 20240924114141.png]]

> Kernel gaussiano de 41x41

![[Pasted image 20240924114332.png]]

> Kernel gaussiano de 500x500

## Superficie Deformante

![[Pasted image 20240924114510.png]]

> a = 20

![[Pasted image 20240924114558.png]]

> a = 60

![[Pasted image 20240924114728.png]]

> a = 40, spiral_step = 20, linewidth = 15

## Efecto cristal

![[Pasted image 20240924114955.png]]

> block_size = 20

![[Pasted image 20240924115035.png]]

> block_size = 60

# Conclusiones

A lo largo de este estudio, hemos explorado tres transformaciones geométricas clave que pueden aplicarse a las imágenes utilizando la biblioteca OpenCV: el suavizado gaussiano, la superficie deformante y el efecto cristal.

El suavizado gaussiano ha demostrado ser una técnica eficaz para reducir el ruido y suavizar los detalles de una imagen, preservando al mismo tiempo rasgos importantes. La implementación paralela de esta transformación aprovecha los múltiples núcleos de la CPU, lo que mejora significativamente el rendimiento.

La transformación de superficie deformante permite distorsionar la imagen de manera controlada, creando interesantes efectos de ondulación y deformación. Esto se logra mediante el cálculo de los gradientes de una matriz en espiral, que luego se utilizan para generar los mapas de desplazamiento necesarios para la deformación.

Por último, el efecto cristal simula la apariencia de una imagen vista a través de una superficie de cristal texturizada. Esto se consigue mediante un patrón de desplazamiento de los píxeles, que genera la ilusión de una distorsión causada por el cristal.

En conjunto, estas transformaciones geométricas ofrecen una amplia gama de posibilidades para mejorar, editar y crear efectos visuales interesantes en las imágenes. Su implementación en OpenCV demuestra la flexibilidad y el poder de esta biblioteca en el campo del procesamiento digital de imágenes.

A medida que los requisitos de los proyectos evolucionan, estas técnicas pueden adaptarse y combinarse para satisfacer diversas necesidades, desde la mejora de la calidad de las imágenes hasta la creación de efectos especiales para la industria del entretenimiento.
