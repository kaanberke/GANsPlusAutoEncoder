from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from keras.datasets import mnist
from keras.models import Model
from sklearn.model_selection import train_test_split


class Autoencoder(object):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder_model_path = 'best_model_encoder.h5'
        self.decoder_model_path = 'best_model_decoder.h5'
        self.patience = 50
        self.log_dir = f'logs/{time()}'
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience, verbose=1)
        self.mc_encoder = ModelCheckpoint(self.encoder_model_path, monitor='val_loss', mode='min', save_best_only=True,
                                     verbose=1)
        self.mc_decoder = ModelCheckpoint(self.decoder_model_path, monitor='val_loss', mode='min', save_best_only=True,
                                     verbose=1)
        self.tensorboard = TensorBoard(log_dir=self.log_dir)
    def __str__(self):
        pass
    def train_encoder(self, x_train, x_test):
        train_length = len(x_train)
        test_length = len(x_test)

        h, w = x_train.shape[1:3]
        try:
            c = x_train.shape[3]
        except:
            c = 1

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = np.reshape(x_train, (train_length, h, w, c))
        x_test = np.reshape(x_test, (test_length, h, w, c))

        self.encoder = Sequential()
        self.encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(h, w, c)))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.encoder.add(UpSampling2D((2, 2)))
        self.encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.encoder.add(UpSampling2D((2, 2)))
        self.encoder.add(Conv2D(16, (3, 3), activation='relu'))
        self.encoder.add(UpSampling2D((2, 2)))
        self.encoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

        self.encoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy',
                        metrics=['mae', 'acc'])

        # DO NOT TRAIN IF THERE IS ONE WHICH IS TRAINED BEFORE
        if os.path.exists(self .encoder_model_path):
            self.encoder.load_weights(self.encoder_model_path)
        else:
            print(self.encoder.summary())
            encoder_history = self.encoder.fit(x_train, x_train,
                                          epochs=500,
                                          validation_data=(x_test, x_test),
                                          shuffle=True,
                                          verbose=1,
                                          callbacks=[self.es, self.mc_encoder, self.tensorboard])
        self.encoder = Model(inputs=self.encoder.input,
                        outputs=self.encoder.layers[5].output)

    def predict_encoder(self, data):
        data = data.reshape(*data.shape, 1)
        encoder_predicted = self.encoder.predict(data)
        return encoder_predicted

    def train_decoder(self, encoder_predicted, x_test):
        h, w = encoder_predicted.shape[1:3]
        try:
            x_c = x_test.shape[3]
        except:
            x_c = 1

        x_test = x_test.astype('float32') / 255.
        x_test = np.reshape(x_test, (*x_test.shape, x_c))

        self.decoder = Sequential()
        self.decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=encoder_predicted.shape[1:]))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2D(16, (3, 3), activation='relu'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

        self.decoder.compile(optimizer='adadelta',
                             loss='binary_crossentropy',
                             metrics=['mae', 'acc'])

        # DO NOT TRAIN IF THERE IS ONE WHICH IS TRAINED BEFORE
        if os.path.exists(self.decoder_model_path):
            self.decoder.load_weights(self.decoder_model_path)
        else:
            print(self.decoder.summary())
            decoder_history = self.decoder.fit(encoder_predicted, x_test,
                                               epochs=10000,
                                               validation_data=(encoder_predicted[:-10], x_test[:-10]),
                                               shuffle=True,
                                               verbose=1,
                                               callbacks=[self.es, self.mc_decoder, self.tensorboard])
    def predict_decoder(self, data):
        decoder_predicted = self.decoder.predict(data)
        return decoder_predicted

    # PLOTTING REAL/ENCODED/DECODED IMAGES
    def plot(self, img, title, path=None, show=True, save_image=False, cmap='gray'):
        img = img.reshape(img.shape[0], img.shape[1])
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        if save_image:
            plt.savefig(path)
        if show:
            plt.show()

from mlxtend.data import loadlocal_mnist
X, y = loadlocal_mnist(
        images_path='./data/train-images.idx3-ubyte',
        labels_path='./data/train-labels.idx1-ubyte')
X = X.reshape(-1, 28, 28)
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=True)
clf = Autoencoder()
clf.train_encoder(X_train, X_test)
encoder_predicted = clf.predict_encoder(X_test)
clf.train_decoder(encoder_predicted, X_test)
decoder_predicted = clf.predict_decoder(encoder_predicted)
clf.plot(decoder_predicted[1], 'decoded')
clf.plot(encoder_predicted[1].reshape(16,8), 'encoded')
clf.plot(X_test[1], 'real')

"""

mkdir data
cd data
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip t*-ubyte.gz

"""