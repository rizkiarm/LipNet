from tensorflow.keras.layers import Conv3D, ZeroPadding3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lipnet.core.layers import CTC
from tensorflow.compat.v1.keras import backend as K


class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28):
        K.set_session
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.conv1)
        self.drop1 = Dropout(0.5)(self.maxp1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.drop1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.conv2)
        self.drop2 = Dropout(0.5)(self.maxp2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.drop2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.conv3)
        self.drop3 = Dropout(0.5)(self.maxp3)

        self.resh1 = TimeDistributed(Flatten())(self.drop3)

        self.lstm_1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1'), merge_mode='concat')(self.resh1)
        self.lstm_2 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'), merge_mode='concat')(self.lstm_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.lstm_2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])
