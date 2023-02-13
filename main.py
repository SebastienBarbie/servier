from model import ModelFromScratch
from dataset import Dataset
import os
import sys
import tensorflow as tf

def main():
    path = 'd:/servier/Test_Technique_Image/Neuroflux_disorder/'
    classes = os.listdir(path)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


    train = Dataset(path, classes, "training", 0.2)
    val = Dataset(path, classes, "validation", 0.2)

    # train = tf.keras.utils.image_dataset_from_directory(
    #     path,
    #     labels='inferred',
    #     label_mode='categorical',
    #     class_names=classes,
    #     batch_size=32,
    #     image_size=(32, 32),
    #     seed=123,
    #     validation_split=0.2,
    #     subset="training",
    #     interpolation='bilinear',
    # )

    # val = tf.keras.utils.image_dataset_from_directory(
    #     path,
    #     labels='inferred',
    #     label_mode='categorical',
    #     class_names=classes,
    #     batch_size=32,
    #     image_size=(32, 32),
    #     seed=123,
    #     validation_split=0.2,
    #     subset="training",
    #     interpolation='bilinear',
    # )
    train.preprocess()

    model = ModelFromScratch()

    model.compile()
    fit = model.fit(train.get_datas(), val.get_datas(), 100)

if __name__ == '__main__':
    sys.exit(main())

