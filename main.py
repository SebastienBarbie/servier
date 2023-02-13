from model import ModelFromScratch
from dataset import Dataset
import os
from tkinter.filedialog import askopenfilename
from tensorflow.keras.preprocessing import image
import numpy as np

def main():

    value = 1
    while value >= 1 and value <= 3:
        value = input("1 - Train the model\n2 - Evaluate the model\n3 - Predict the stage of the disease\n4 - Exit the programm\n")
        if value.isdigit():
            value = int(value)
        else:
            value = 0

        path = 'd:/servier/Test_Technique_Image/Neuroflux_disorder/'
        classes = os.listdir(path)
        model = ModelFromScratch()

        if value == 1 :
            epochs = input("How many epochs ?\n")
            if epochs.isdigit():
                epochs = int(epochs)
            else:
                epochs = 10

            
        
            classes = os.listdir(path)

            train = Dataset(path, classes, "training", 0.2)
            train.preprocess()
            train.generate_coeffs()

            print(train.coeffs)

            val = Dataset(path, classes, "validation", 0.2)

            model.fit(train.get_datas(), val.get_datas(), epochs, train.coeffs)

        if value == 2:
            train = Dataset(path, classes, "training", 0.2)
            train.preprocess()
            train.generate_coeffs()
            model = ModelFromScratch()
            model.evaluate(train.get_datas())

        if value == 3: 
            classes = os.listdir(path)
            FILETYPES = [ ("jpg files", "*.jpg") ]
            file = askopenfilename(filetypes=FILETYPES)
            img = image.load_img(file, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            predict = model.predict(img_batch)
            print(classes[predict.argmax()])

if __name__ == '__main__':
    main()

