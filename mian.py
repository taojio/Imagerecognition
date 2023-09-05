import os
import numpy as np
from keras import layers
from keras import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import layers, models
import torch
from torch import nn

# 定义文件夹地址
folder_path = 'D:/AI/Cu'
# 读取当前文件夹名
folder_name = os.listdir(folder_path)
#提取上级文件名称
parent_folder_name = os.path.basename(os.path.normpath(folder_path))
# 打印上级文件名称
print(parent_folder_name)
#读取训练集和测试集npy文件
train_data = np.load(os.path.join(folder_path, parent_folder_name + '_1.jpg' + '_train.npy'))
test_data = np.load(os.path.join(folder_path, parent_folder_name + '_1.jpg' + '_test.npy'))

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))
#定义反向传播
def grads_and_vars(x):
    return model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
#定义Adam优化器
optimizer = keras.optimizers.Adam(learning_rate=0.001)

#定义损失函数
def categorical_crossentropy(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#定义准确率
def accuracy(y_true, y_pred):
    return tf.keras.metrics.accuracy(y_true, y_pred)

#编译模型
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
#打印模型结构
print(model.summary())
#定义超参数
epochs = 1000
batch_size = 28
#定义学习率
learning_rate = 0.002 # 将学习率从0.01改为0.001
#定义训练步长
step_size = 1000 # 将步长从1000改为100

#训练模型
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#保存模型为ckpt格式文件
model.save('model.ckpt')     # The model will be saved to the current directory as "model"

