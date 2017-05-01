from lipnet.lipreading.helpers import text_to_labels
from lipnet.lipreading.videos import Video
from lipnet.lipreading.aligns import Align
from lipnet.helpers.threadsafe import threadsafe_generator
from lipnet.helpers.list import get_list_safe
from keras import backend as K
import numpy as np
import keras
import pickle
import os
import glob
import multiprocessing


# datasets/[train|val]/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
class BasicGenerator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=30,
                 vtype='mouth', face_predictor_path=None, curriculum=None, start_epoch=0):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.current_epoch  = multiprocessing.Value('i', start_epoch)
        self.curriculum     = curriculum
        if self.curriculum is not None:
            self.update_curriculum()
        self.vtype               = vtype
        self.face_predictor_path = face_predictor_path

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def validation_size(self):
        return len(self.val_list)

    def get_output_size(self):
        return 28

    def get_cache_path(self):
        return self.dataset_path.rstrip('/') + '.cache'

    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video(self.vtype, self.face_predictor_path).from_video(video_path)
                else:
                    video = Video(self.vtype, self.face_predictor_path).from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print "Error loading video: "+video_path
                continue
            if K.image_data_format() == 'channels_first' and video.data.shape != (self.img_c,self.frames_n,self.img_w,self.img_h):
                print "Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.img_c,self.frames_n,self.img_w,self.img_h))+""
                continue
            if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n,self.img_w,self.img_h,self.img_c):
                print "Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.frames_n,self.img_w,self.img_h,self.img_c))+""
                continue
            video_list.append(video_path)
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('/')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        if os.path.isfile(self.get_cache_path()):
            print "\nLoading dataset list from cache..."
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
        else:
            print "\nEnumerating dataset list from disk..."
            self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
            self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print "Found {} videos for training.".format(self.training_size)
        print "Found {} videos for validation.".format(self.validation_size)
        print ""

        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list

        X_data_path = get_list_safe(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        for path in X_data_path:
            video = Video().from_frames(path)
            align = self.get_align(path.split('/')[-1])
            video_unpadded_length = video.length
            if self.curriculum is not None:
                video, align, video_unpadded_length = self.curriculum.apply(video, align)
            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append([align.label_length])
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append([video.length]) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence)

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        while 1:
            if self.curriculum is not None and self.curriculum.epoch != self.current_epoch.value:
                self.update_curriculum(train=True)
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.training_size:
                self.cur_train_index = self.cur_train_index % self.minibatch_size
                np.random.shuffle(self.train_list)
            yield ret

    @threadsafe_generator
    def next_val(self):
        while 1:
            if self.curriculum is not None and self.curriculum.epoch != self.current_epoch.value:
                self.update_curriculum(train=False)
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.validation_size:
                self.cur_val_index = self.cur_val_index % self.minibatch_size
            yield ret

    def on_train_begin(self, logs={}):
        self.cur_train_index = 0
        self.cur_val_index = 0

    def on_epoch_begin(self, epoch, logs={}):
        with self.current_epoch.get_lock():
            self.current_epoch.value = epoch

    def update_curriculum(self, train=True):
        self.curriculum.update(self.current_epoch.value, train=train)
        print "Epoch {}: {}".format(self.current_epoch.value, self.curriculum)


# datasets/video/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
class RandomSplitGenerator(BasicGenerator):
    def build(self, **kwargs):
        self.video_path = os.path.join(self.dataset_path, 'video')
        self.align_path = os.path.join(self.dataset_path, 'align')
        self.val_split = kwargs.get('val_split', 0.2)
        self.build_dataset()
        return self
        
    def build_dataset(self):
        if os.path.isfile(self.get_cache_path()):
            print "\nLoading dataset list from cache..."
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
        else:
            print "\nEnumerating dataset list from disk..."
            video_list = self.enumerate_videos(os.path.join(self.video_path, '*', '*'))
            np.random.shuffle(video_list) # Random the video list before splitting
            if(self.val_split > 1): # If val_split is not a probability
                training_size = len(video_list) - self.val_split
            else: # If val_split is a probability
                training_size = len(video_list) - int(self.val_split * len(video_list))
            self.train_list = video_list[0:training_size]
            self.val_list   = video_list[training_size:]
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print "Found {} videos for training.".format(self.training_size)
        print "Found {} videos for validation.".format(self.validation_size)
        print ""