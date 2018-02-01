#! -*- coding: utf-8 -*-

from keras import backend as K
from keras import utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model

from Capsule_Keras import *
from config import *

# 准备训练数据
from test_preprocess import embedding_matrix_path
from util import preprocessing

batch_size = 128

data, X_test, y, word_index = preprocessing.load_train_test_y()
x_train = data[:-SPLIT]
y_train = y[:-SPLIT]
print(x_train.shape, y_train.shape)

x_test = data[-SPLIT:]
y_test = y[-SPLIT:]
print(x_test.shape, y_test.shape)

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16


def get_model(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            # weights=[embedding_matrix],
                            trainable=False)(input1)
    x = Bidirectional(GRU(gru_len, activation='relu', return_sequences=True))(embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

embedding_matrix1 = np.load(embedding_matrix_path)

model=get_model(embedding_matrix1)
bst_model_path = 'capsule' + '.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,verbose=1,  save_weights_only=True)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, model_checkpoint])
