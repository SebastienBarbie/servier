import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, Layer, BatchNormalization, InputLayer
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

class ModelFromScratch():
    def __init__(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

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

    def fit(self, train, val, epochs):
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
        fit = self._model.fit(train,epochs=epochs, validation_data=val)#, class_weight = coeff,callbacks=[mcp_save, reduce_lr_loss])
    
    def evaluate(self, ds):
        self._model.evaluate(ds)
    
    def get_model(self):
        return self._model
    
