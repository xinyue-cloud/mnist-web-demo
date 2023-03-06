import datetime
import glob
import os
from os.path import basename

import numpy as np

import keras
from PIL import Image
from keras import layers, callbacks
from keras.datasets import mnist
from keras.utils import to_categorical

from lib.timer import timer
import matplotlib.pyplot as plt

model_h5_file = 'saved/model.h5'
model_json_file = 'saved/model.json'


def append_local_data(data, labels):
    # 将本地存储的训练数据添加至训练集
    new_images_dir = 'train-images'
    new_images_paths = glob.glob(os.path.join(new_images_dir, "*.png"))
    if len(new_images_paths) > 0:
        new_images = []
        new_labels = []
        for image_path in new_images_paths:
            with Image.open(image_path) as img:
                img = img.convert('L')
                img = img.resize((28, 28))
                img = np.array(img) / 255.0
                new_images.append(img)
                # 获取新图像的相应标签
                label = int(basename(image_path).split('_')[0])
                new_labels.append(label)

        new_images = np.array(new_images)
        new_labels = np.array(new_labels)

        # 向原始数据集添加新的图像和标签
        data = np.concatenate((data, new_images))
        labels = np.concatenate((labels, new_labels))
    return data, labels


@timer
def start_train():
    # 导入示例数据库
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    (train_data, train_labels) = append_local_data(train_data, train_labels)

    # 将数值缩放至0-1之间
    train_data = train_data / 255.
    test_data = test_data / 255.

    # 转换为模型可以理解的数据形式.
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 构建模型, 激活函数 relu 表示计算的特征值小于0时返回0
    model = keras.Sequential([

        # 第一种训练模型

        # 卷积层: 32个卷积核,卷积核大小为(3,3),激活函数为relu,输入张量的形状为(28,28,1)
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        # 池化层: 池化窗口的大小为(2,2),取其中最大的值作为输出.
        layers.MaxPool2D((2, 2)),
        # 卷积层: 32个卷积核,卷积核大小为(3,3),激活函数为relu
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层: 池化窗口的大小为(2,2)
        layers.MaxPool2D((2, 2)),
        # 数据进行了形状变换,从多维数据转换为一维数组
        layers.Flatten(),
        # 全连接层: 64个神经元,激活函数为relu
        layers.Dense(64, activation='relu'),
        # 全连接层: 10个神经元,激活函数为softmax
        layers.Dense(10, activation='softmax')

        # 第二种训练模型
        # layers.Conv2D(32, (5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)),
        # layers.Conv2D(32, (5, 5), padding='Same', activation='relu'),
        # layers.MaxPool2D((2, 2)),
        # layers.Dropout(0.25),
        # layers.Conv2D(64, (3, 3), padding='Same', activation='relu'),
        # layers.Conv2D(64, (3, 3), padding='Same', activation='relu'),
        # layers.MaxPool2D((2, 2), (2, 2)),
        # layers.Dropout(0.25),
        # layers.Flatten(),
        # layers.Dense(256, activation='relu'),
        # layers.Dropout(0.25),
        # layers.Dense(10, activation='softmax')

    ])

    # 编译及训练
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 输出训练日志，可使用 tensorboard --logdir logs/fit 进行查看
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    result = model.fit(train_data, train_labels, epochs=10, batch_size=128,
                       validation_data=(test_data, test_labels),
                       callbacks=[tensorboard_callback])
    history = result.history
    print("history_key: ", history.keys())

    # 输出“训练损失与验证损失”图
    loss_values = history['loss']
    val_loss_values = history['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/epochs-loss.png')
    # plt.show()

    # 输出“训练损失与验证损失”图
    acc_values = history['accuracy']
    val_acc_values = history['val_accuracy']
    plt.clf()
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/epochs-acc.png')
    # plt.show()

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

    # 保存训练结果
    model.save(model_h5_file)

    return model
