import numpy as np
from lipnet.lipreading.videos import VideoAugmenter

class Curriculum(object):
    def __init__(self, rules):
        self.rules = rules

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.update()

    def update(self):
        current_rule = self.rules(self.epoch)
        self.sentence_length = current_rule.get('sentence_length') or -1
        self.flip_probability = current_rule.get('flip_probability') or 0.0
        self.jitter_probability = current_rule.get('jitter_probability') or 0.0

    def apply(self, video, align):
        original_video = video
        if self.sentence_length > 0:
            video, align = VideoAugmenter.pick_subsentence(video, align, self.sentence_length)
        if np.random.ranf() < self.flip_probability:
            video = VideoAugmenter.horizontal_flip(video)
        video = VideoAugmenter.temporal_jitter(video, self.jitter_probability)
        video_unpadded_length = video.length
        video = VideoAugmenter.pad(video, original_video.length)
        return video, align, video_unpadded_length

    def __str__(self):
        return "{}(sentence_length: {}, flip_probability: {}, jitter_probability: {})"\
            .format(self.__class__.__name__, self.sentence_length, self.flip_probability, self.jitter_probability)