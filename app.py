import base64
import io
import os.path
import re

import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request
from keras.datasets import mnist
from keras.utils import img_to_array, load_img

from train import model_h5_file, start_train

NOT_TRAIN = os.path.exists(model_h5_file)

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if NOT_TRAIN:
    model = keras.models.load_model(model_h5_file)
else:
    model = start_train()

print('Number of layers:', len(model.layers))


def parse_image(img_data):
    img_str = re.search(b'base64,(.*)', img_data).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(img_str))


def random_train_images():
    print('random_train_images')
    # 从训练数据中随机获取50张图像
    num_images = 400
    idx = np.random.choice(train_images.shape[0], num_images, replace=False)
    images = train_images[idx]

    # 合成图片
    rows = 10
    cols = 40
    height, width = images[0].shape
    combined_image = np.zeros((height * rows, width * cols))

    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            if k < num_images:
                combined_image[i * height:(i + 1) * height, j * width:(j + 1) * width] = images[k]

    # 将Numpy数组转换为PIL图像
    combined_image = combined_image.astype(np.uint8)
    image = Image.fromarray(combined_image)
    image.save('static/random_images.png')


def random_test_image():
    # 随机选择一张测试图像
    idx = np.random.choice(test_images.shape[0], 1)
    image = test_images[idx][0]

    # 将图像尺寸放大至280
    image = Image.fromarray(image)
    image = image.resize((280, 280))

    # 保存图像
    image.save('static/test_image.png')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/test-image')
def rando_image():
    # 随机选择一张测试图像
    idx = np.random.choice(test_images.shape[0], 1)
    image = test_images[idx][0]
    # 将图像尺寸放大至280
    image = Image.fromarray(image)
    image = image.resize((280, 280))
    # 将图像保存到内存中
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    # 将图像数据转换为Blob
    return Response(img_io.getvalue(), mimetype='image/png')


@app.route('/predict', methods=['Get', 'POST'])
def predict():
    parse_image(request.get_data())
    img = img_to_array(load_img('output.png', target_size=(28, 28), color_mode="grayscale")) / 255.
    img = np.expand_dims(img, axis=0)
    # img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)

    exp_pred = np.exp(prediction)
    sum_exp = np.sum(exp_pred)
    percentage_pred = exp_pred / sum_exp
    percentage_str = ["{:.2%}".format(p) for p in percentage_pred[0]]

    label = np.argmax(prediction)
    response = {'prediction': percentage_str, 'label': int(label)}
    print('response: ', response)
    return jsonify(response)


if __name__ == '__main__':
    random_train_images()
    app.run(host="0.0.0.0", port=4345, debug=True)
