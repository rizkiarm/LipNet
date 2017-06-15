from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os
import glob

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

lipnet = None
adam = None
spell = None
decoder = None

def predict(weight_path, video):
    global lipnet
    global adam
    global spell
    global decoder

    if lipnet is None:
        lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=75,
                        absolute_max_string_len=32, output_size=28)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.model.load_weights(weight_path)

        spell = Spell(path=PREDICT_DICTIONARY)
        decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                          postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    y_pred         = lipnet.predict(X_data)
    result         = decoder.decode(y_pred, input_length)[0]

    show_video_subtitle(video.face, result)
    print result

def predicts(weight_path, videos_path, absolute_max_string_len=32, output_size=28):
    videos = []
    for video_path in glob.glob(os.path.join(videos_path, '*')):
        videos.append(load(video_path))
    raw_input("Press Enter to continue...")
    for video in videos:
        predict(weight_path, video)

def load(video_path):
    print "\n[{}]\nLoading data from disk...".format(video_path)
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"
    return video

if __name__ == '__main__':
    if len(sys.argv) == 3:
        predicts(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        predicts(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        predicts(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        pass