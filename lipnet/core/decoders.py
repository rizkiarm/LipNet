from lipnet.core.search import ctc_beamsearch
import itertools
import numpy as np

def decode_batch_best(test_func, output_batch, decoder_func=lambda x: x):
    out = test_func([output_batch, 0])[0] # the first 0 indicates test
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        ret.append(decoder_func(out_best))
    return ret

def decode_batch_beam(test_func, output_batch, alphabets='abcdefghijklmnopqrstuvwxyz -', k=5):
    out = test_func([output_batch, 0])[0] # the first 0 indicates test
    ret = []
    for j in range(out.shape[0]):
        probdistb = out[j, 2:]
        dist = np.log(probdistb)
        out_ctcbeam = ctc_beamsearch(dist, alphabets, k=k).replace("-", "")
        ret.append(out_ctcbeam)
    return ret

def decode_batch_beam_lang(test_func, output_batch, decoder_func=lambda x: x, k=5, lang=None):
	pass