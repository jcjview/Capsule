from keras import backend as K
from keras import utils
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model

from Capsule_Keras import *
from capsule_test import Transpose
from config import *

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


gru_len=128
Routings=5
Num_capsule=10
Dim_capsule=16

input1 = Input(shape=(MAX_TEXT_LENGTH,))
embed_layer = Embedding(MAX_FEATURES,
                        embedding_dims,
                        input_length=MAX_TEXT_LENGTH,
                        # weights=[embedding_matrix],
                        trainable=False)(input1)
x = Bidirectional(GRU(gru_len, activation='relu', return_sequences=True))(embed_layer)

# x = Bidirectional(GRU(gru_len, activation='relu',return_sequences=True))(x)
# x = Transpose(perm=[0, 2, 1])(x)
# cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
# cnn = AveragePooling2D((2, 2))(cnn)
# cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
# cnn = Conv2D(128, (3, 3), activation='relu')(cnn)

# x = Reshape((0, 0, 0, 1))(x)
capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                  share_weights=True)(x)
# output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
output=Dense(6, activation='sigmoid')(capsule)
model = Model(inputs=input1, outputs=output)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
model.summary()