from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from keras.datasets import mnist
from keras.models import Model

encoder_model_path = 'best_model_encoder.h5'
decoder_model_path = 'best_model_decoder.h5'

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1)
mc_encoder = ModelCheckpoint(encoder_model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
mc_decoder = ModelCheckpoint(decoder_model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=f'logs/{time()}')

encoder = Sequential()
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(4, 4, 8)))
encoder.add(UpSampling2D((2, 2)))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(UpSampling2D((2, 2)))
encoder.add(Conv2D(16, (3, 3), activation='relu'))
encoder.add(UpSampling2D((2, 2)))
encoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

encoder.compile(optimizer='adadelta',
                loss='binary_crossentropy',
                metrics=['mae', 'acc'])

if os.path.exists(encoder_model_path):
    encoder.load_weights(encoder_model_path)

print(encoder.summary())
encoder_history = encoder.fit(x_train, x_train,
                              epochs=300,
                              validation_data=(x_test, x_test),
                              shuffle=True,
                              verbose=1,
                              callbacks=[es, mc_encoder, tensorboard])
encoder = Model(inputs=encoder.input,
                outputs=encoder.layers[5].output)
encoder_predicted = encoder.predict(x_test)

decoder = Sequential()
decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(4, 4, 8)))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

decoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['mae', 'acc'])

if os.path.exists(decoder_model_path):
    decoder.load_weights(decoder_model_path)

print(decoder.summary())
decoder_history = decoder.fit(encoder_predicted[:-10], x_test[:-10],
                              epochs=300,
                              validation_data=(encoder_predicted[:-10], x_test[:-10]),
                              shuffle=True,
                              verbose=1,
                              callbacks=[es, mc_decoder, tensorboard])

index = -5
predicted_encoded = encoder.predict(x_test[index].reshape(1, 28, 28, 1))
predicted_decoded = decoder.predict(encoder_predicted[index].reshape(1, 4, 4, 8))

plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title('Real')
plt.axis('off')
plt.show()

plt.imshow(predicted_encoded.reshape(16, 8), cmap='gray')
plt.title('Encoded')
plt.axis('off')
plt.show()

plt.imshow(predicted_decoded.reshape(28, 28), cmap='gray')
plt.title('Decoded')
plt.axis('off')
plt.show()

