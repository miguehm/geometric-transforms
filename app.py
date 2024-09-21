from flask import Flask, jsonify, request, send_file
import numpy as np
from PIL import Image
from gaussian_blur import gaussian_blur
from deforming_surface import deforming_surface_spiral
import io


app = Flask(__name__)


@app.route('/')
def home():
    return "esto es una prueba"


# Messages
matrix_error = "Matrix multiplication requires two 2D arrays"


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    num1 = np.array(data.get('num1'))
    num2 = np.array(data.get('num2'))
    operation = data.get('operation')

    if operation == 'add':
        result = np.add(num1, num2)
    elif operation == 'subtract':
        result = np.subtract(num1, num2)
    elif operation == 'multiply':
        # Si num1 y num2 son matrices, realiza la multiplicación de matrices
        if len(num1.shape) > 1 and len(num2.shape) > 1:
            result = np.matmul(num1, num2)
        else:
            result = np.multiply(num1, num2)
    elif operation == 'divide':
        result = np.divide(num1, num2)
    elif operation == 'matrix_multiply':
        if len(num1.shape) == 2 and len(num2.shape) == 2:
            result = np.matmul(num1, num2)
        else:
            return jsonify({'error': matrix_error}), 400
    else:
        return jsonify({'error': 'Invalid operation'}), 400

    return jsonify({'result': result.tolist()})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    elements = request.form.get('gauss_size', 1)

    try:
        elements = int(elements)
    except ValueError:
        return jsonify({'error': 'El filtro Gaussiano debe ser entero'}), 400

    # Open the image file
    image = Image.open(file)

    # pasar a array de numpy
    image = np.array(image)

    processed_image = gaussian_blur(image, elements)

    img_io = io.BytesIO()
    Image.fromarray(processed_image).save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the image
    return send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='processed_image.jpg')


@app.route('/image_to_spiral_transform', methods=['POST'])
def image_to_spiral_transform():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    step = request.form.get('step', 10)
    line_width = request.form.get('line-width', 30)
    a = request.form.get('a', 41)

    try:
        step = int(step)
        line_width = int(line_width)
        a = int(a)

        # Open the image file
        image = Image.open(file)

        # pasar a array de numpy
        image = np.array(image)

        if (image.shape[0] != image.shape[1]):
            raise ValueError('La imagen debe ser cuadrada')

    except ValueError:
        return jsonify({'error': 'Los valores deben ser enteros y la imagen cuadrada'}), 400

    processed_image = deforming_surface_spiral(image, step, line_width, a)

    img_io = io.BytesIO()
    Image.fromarray(processed_image).save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the image
    return send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='processed_image.jpg')


if __name__ == '__main__':
    app.run(debug=True)
