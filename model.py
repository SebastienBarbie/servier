import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, Layer, BatchNormalization, InputLayer
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os


class ModelFromScratch():
    def __init__(self, path='./model'):
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        try :
            model = keras.models.load_model(path)


        except :
            model = Sequential()
            model.add(Rescaling(1./255))

            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())

            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.15))


            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())

            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.15))

            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())


            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.15))


            model.add(Flatten())

            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))

        self._model = model

    def compile(self):
        self._model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', 'Recall'])

    def fit(self, train, val, epochs, coeff):
        self._model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', 'Recall'])
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

        # earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
        fit = self._model.fit(train,epochs=epochs, validation_data=val, class_weight = coeff)# ,callbacks=[mcp_save, reduce_lr_loss])
        self._model.save('./model')

    
    def evaluate(self, ds):
        self._model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', 'Recall'])
        self._model.evaluate(ds)
    
    def get_model(self):
        return self._model

    def save(self):
        self._model.save(os.absolutpath('./model'))

    def predict(self, x):
        return self._model.predict(x)
    
