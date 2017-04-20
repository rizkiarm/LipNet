from keras.optimizers import Adam
from lipnet.lipreading.generators import BasicGenerator
from lipnet.lipreading.callbacks import Statistics
from lipnet.model import LipNet
import numpy as np
import sys

np.random.seed(55)

def stats(weight_path, dataset_path, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
	lip_gen = BasicGenerator(dataset_path=dataset_path, 
                                minibatch_size=minibatch_size,
                                img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                absolute_max_string_len=absolute_max_string_len).build()

	lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, 
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())

	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
	lipnet.model.load_weights(weight_path)

	statistics  = Statistics(lipnet.test_function, lip_gen.next_val(), 256, output_dir=None)

	lip_gen.on_train_begin()
	statistics.on_epoch_end(0)

if __name__ == '__main__':
	weight_path = sys.argv[1]
	dataset_path = sys.argv[2]
	stats(weight_path, dataset_path, 3, 100, 50, 75, 32, 50)