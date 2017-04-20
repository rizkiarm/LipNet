from lipnet.lipreading.curriculums import Curriculum
from lipnet.lipreading.videos import Video
from lipnet.lipreading.aligns import Align
from lipnet.lipreading.helpers import text_to_labels
from lipnet.lipreading.visualization import show_video_subtitle

def rules(epoch):
    if epoch == 0:
        return {'sentence_length': 1, 'flip_probability': 0, 'jitter_probability': 0}
    if epoch == 1:
        return {'sentence_length': 2, 'flip_probability': 0.5, 'jitter_probability': 0}
    if epoch == 2:
        return {'sentence_length': 3, 'flip_probability': 0.5, 'jitter_probability': 0.05}
    if epoch == 3:
        return {'sentence_length': -1, 'flip_probability': 0, 'jitter_probability': 0}
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}

curriculum = Curriculum(rules)

video = Video(vtype='face', face_predictor_path='evaluation/models/shape_predictor_68_face_landmarks.dat')
video.from_video('evaluation/samples/id2_vcd_swwp2s.mpg')

align = Align(absolute_max_string_len=32, label_func=text_to_labels).from_file('evaluation/samples/swwp2s.align')

for i in range(5):
    curriculum.set_epoch(i)
    print curriculum
    _video, _align, _ = curriculum.apply(video, align)
    show_video_subtitle(frames=_video.face, subtitle=_align.sentence)