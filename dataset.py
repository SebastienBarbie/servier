import tensorflow as tf
import os
from keras.models import Sequential
from tensorflow.keras import layers


class Dataset:
    def __init__(self, path, classes, substet, validation_split):
        self.datas = tf.keras.utils.image_dataset_from_directory(
                path,
                labels='inferred',
                label_mode='categorical',
                class_names=classes,
                batch_size=32,
                image_size=(32, 32),
                seed=123,
                validation_split=validation_split,
                subset=substet,
                interpolation='bilinear',
        )

    def preprocess(self):
        flip = Sequential(
            [
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
            ]
        )
        rot = Sequential(
            [
                layers.RandomRotation(0.5),
                layers.RandomRotation(factor=0.15),
                layers.RandomZoom(0.2, 0.3),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
            ]
        )

        rot2 = Sequential(
            [
                layers.RandomRotation(0.5),
                layers.RandomZoom((-0.2, -0.3)),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.8),
            ]
        )

        augmented_rot = self.datas.map(
        lambda x, y: (rot(x, training=True), y))

        augmented_rot2 = self.datas.map(
        lambda x, y: (rot2(x, training=True), y))

        augmented_flip = self.datas.map(
        lambda x, y: (flip(x, training=True), y))

        self.datas.concatenate(augmented_flip)
        self.datas.concatenate(augmented_rot)
        # self.datas.concatenate(rot2)
    
    def get_datas(self):
        return self.datas