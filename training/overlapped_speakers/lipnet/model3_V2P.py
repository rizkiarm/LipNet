from tensorflow.keras.layers import Conv3D, ZeroPadding3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import LSTM
from keras_contrib.layers import GroupNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lipnet.core.layers import CTC
from tensorflow.compat.v1.keras import backend as K


class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=74, absolute_max_string_len=32, output_size=28):
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

        self.zero1 = ZeroPadding3D(padding=(1, 0, 0), name='zero1')(self.input_data)
        self.conv1 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.gn1 = GroupNormalization(name='gn1', groups=32)(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.gn1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.actv1)

        self.zero2 = ZeroPadding3D(padding=(1, 0, 0), name='zero2')(self.maxp1)
        self.conv2 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.gn2 = GroupNormalization(name='gn2', groups=32)(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.gn2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.actv2)

        self.zero3 = ZeroPadding3D(padding=(1, 0, 0), name='zero3')(self.maxp2)
        self.conv3 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.gn3 = GroupNormalization(name='gn3', groups=32)(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.gn3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.actv3)

        self.zero4 = ZeroPadding3D(padding=(1, 0, 0), name='zero4')(self.maxp3)
        self.conv4 = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv4')(self.zero4)
        self.gn4 = GroupNormalization(name='gn4', groups=32)(self.conv4)
        self.actv4 = Activation('relu', name='actv4')(self.gn4)

        self.zero5 = ZeroPadding3D(padding=(1, 0, 0), name='zero5')(self.actv4)
        self.conv5 = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv5')(self.zero5)
        self.gn5 = GroupNormalization(name='gn5', groups=32)(self.conv5)
        self.actv5 = Activation('relu', name='actv5')(self.gn5)
        self.maxp5 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), name='max5')(self.actv5)

        self.resh1 = TimeDistributed(Flatten())(self.maxp5)

        self.lstm_1 = Bidirectional(LSTM(768, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1'), merge_mode='concat')(self.resh1)
        self.lstm_1_gn = GroupNormalization(name='lstm_1_gn', groups=32)(self.lstm_1)
        self.lstm_2 = Bidirectional(LSTM(768, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'), merge_mode='concat')(self.lstm_1_gn)
        self.lstm_2_gn = GroupNormalization(name='lstm_2_gn', groups=32)(self.lstm_2)
        self.lstm_3 = Bidirectional(LSTM(768, return_sequences=True, kernel_initializer='Orthogonal', name='lstm3'), merge_mode='concat')(self.lstm_2_gn)
        self.lstm_3_gn = GroupNormalization(name='lstm_3_gn', groups=32)(self.lstm_3)

        # transforms RNN output to character activations:
        self.dense1 = Dense(768, kernel_initializer='he_normal', name='dense1')(self.lstm_3_gn)
        self.gn6 = GroupNormalization(name='gn6', groups=32)(self.dense1)
        self.actv6 = Activation('relu', name='actv6')(self.gn6)
        self.dense2 = Dense(self.output_size, kernel_initializer='he_normal', name='dense2')(self.actv6)


        self.y_pred = Activation('softmax', name='softmax')(self.dense2)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_gnh):
        return self.test_function([input_gnh, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])
