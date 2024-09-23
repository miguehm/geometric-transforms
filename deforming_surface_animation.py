from deforming_surface import deforming_surface_spiral
from cristal import crystal_transform
from PIL import Image
import numpy as np
import cv2
import imageio


def add_text_to_image(image, text):
    (h, w) = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # Blanco
    background_color = (0, 0, 0)  # Fondo Negro

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    text_x = w - text_size[0] - 10
    text_y = h - 10  # 10 píxeles de margen desde el borde inferior

    cv2.rectangle(image,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  background_color,
                  thickness=cv2.FILLED)

    cv2.putText(image, text, (text_x, text_y), font,
                font_scale, text_color, font_thickness)

    return image


def spiral_animation():
    image = Image.open('image.jpg')
    image = np.array(image)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    images = []

    for i in range(1, 60):
        output = deforming_surface_spiral(image, a=i)
        output = add_text_to_image(output, f'a = {i}')
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        images.append(output_rgb)

    for i in range(60, 0, -1):
        output = deforming_surface_spiral(image, a=i)
        output = add_text_to_image(output, f'a = {i}')
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        images.append(output_rgb)

    imageio.mimsave('anim_spiral.gif', images, duration=0.5)


def crystal_animation():
    image = Image.open('image.jpg')
    image = np.array(image)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    images = []

    for i in range(1, 60):
        output = crystal_transform(image, block_size=i)
        output = add_text_to_image(output, f'block_size = {i}')
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        images.append(output_rgb)

    for i in range(60, 0, -1):
        output = crystal_transform(image, block_size=i)
        output = add_text_to_image(output, f'block_size = {i}')
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        images.append(output_rgb)

    imageio.mimsave('anim_cristal.gif', images, duration=0.5)


if __name__ == '__main__':
    # spiral_animation()
    crystal_animation()
