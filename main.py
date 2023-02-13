from model import ModelFromScratch
from dataset import Dataset
import os
import sys
import tensorflow as tf
import keras

def main():

    value = 0
    while value < 1 or value > 3:
        value = input("1 - Train the model\n2 - Evaluate the model\n3 - Predict the stage of the disease\n")
        if value.isdigit():
            value = int(value)
        else:
            value = 0

    
    if value == 1 :
        epochs = input("How many epochs ?\n")
        if epochs.isdigit():
            epochs = int(epochs)
        else:
            epochs = 10

        model = ModelFromScratch()
        path = 'd:/servier/Test_Technique_Image/Neuroflux_disorder/'
        classes = os.listdir(path)

        train = Dataset(path, classes, "training", 0.2)
        train.preprocess()
        train.generate_coeffs()

        print(train.coeffs)

        val = Dataset(path, classes, "validation", 0.2)

        model.fit(train.get_datas(), val.get_datas(), epochs, train.coeffs)
    



    

    # train.preprocess()

    # model = ModelFromScratch()

    # model.compile()
    # fit = model.fit(train.get_datas(), val.get_datas(), 100)

if __name__ == '__main__':
    main()

