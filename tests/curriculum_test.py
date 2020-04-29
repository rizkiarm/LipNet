from lipnet.lipreading.curriculums import Curriculum
from lipnet.lipreading.videos import Video
from lipnet.lipreading.aligns import Align
from lipnet.lipreading.helpers import text_to_labels, labels_to_text
from lipnet.lipreading.visualization import show_video_subtitle
import numpy as np

def rules(epoch):
    if epoch == 0:
        return {'sentence_length': 1, 'flip_probability': 0, 'jitter_probability': 0}
    if epoch == 1:
        return {'sentence_length': 2, 'flip_probability': 0.5, 'jitter_probability': 0}
    if epoch == 2:
        return {'sentence_length': 3, 'flip_probability': 0.5, 'jitter_probability': 0.05}
    if epoch == 3:
        return {'sentence_length': -1, 'flip_probability': 0, 'jitter_probability': 0}
    if epoch == 4:
        return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0}
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}

def show_results(_video, _align, video, align):
    show_video_subtitle(frames=_video.face, subtitle=_align.sentence)
    print ("Video: ")
    print (_video.length)
    print (np.array_equiv(_video.mouth, video.mouth))
    print (np.array_equiv(_video.data, video.data))
    print (np.array_equiv(_video.face, video.face))
    print ("Align: ")
    print (labels_to_text(_align.padded_label.astype(np.int)))
    print (_align.padded_label)
    print (_align.label_length)
    print (np.array_equiv(_align.sentence, align.sentence))
    print (np.array_equiv(_align.label, align.label))
    print (np.array_equiv(_align.padded_label, align.padded_label))

curriculum = Curriculum(rules)

video = Video(vtype='face', face_predictor_path='evaluation/models/shape_predictor_68_face_landmarks.dat')
video.from_video('evaluation/samples/id2_vcd_swwp2s.mpg')

align = Align(absolute_max_string_len=32, label_func=text_to_labels).from_file('evaluation/samples/swwp2s.align')

print ("=== TRAINING ===")
for i in range(6):
    curriculum.update(i, train=True)
    print (curriculum)
    _video, _align, _ = curriculum.apply(video, align)
    show_results(_video, _align, video, align)

print ("=== VALIDATION/TEST ===")
for i in range(6):
    curriculum.update(i, train=False)
    print (curriculum)
    _video, _align, _ = curriculum.apply(video, align)
    show_results(_video, _align, video, align)
