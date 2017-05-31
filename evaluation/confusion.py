import nltk
import sys
import string
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

arpabet = nltk.corpus.cmudict.dict()

def get_phonemes(word):
    return [str(phoneme).translate(None, string.digits) for phoneme in arpabet[word][0]]

with open('phonemes.txt') as f:
    labels = f.read().splitlines()

V1 = labels[0:7]
V2 = labels[7:10]
V3 = labels[10:14]
V4 = labels[14:17]
A = labels[17:21]
B = labels[21:23]
C = labels[23:27]
D = labels[27:31]
E = labels[31:34]
F = labels[34:36]
G = labels[36:38]
H = labels[38:42]

SCENARIOS = [
    ('Phonemes', labels),
    ('Lip-rounding based vowels', V1+V2+V3+V4),
    ('Alveolar-semivowels', A),
    ('Alveolar-fricatives', B),
    ('Alveolar', C),
    ('Palato-alveolar', D),
    ('Bilabial', E),
    ('Dental', F),
    ('Labio-dental', G),
    ('Velar', H)
]

def get_viseme(word):
    phonemes = get_phonemes(word)
    visemes = []
    for phoneme in phonemes:
        if phoneme in V1+V2+V3+V4:
            visemes.append('V')
        elif phoneme in A:
            visemes.append('A')
        elif phoneme in B:
            visemes.append('B')
        elif phoneme in C:
            visemes.append('C')
        elif phoneme in D:
            visemes.append('D')
        elif phoneme in E:
            visemes.append('E')
        elif phoneme in F:
            visemes.append('F')
        elif phoneme in G:
            visemes.append('G')
        elif phoneme in H:
            visemes.append('H')
    return visemes

def get_confusion_matrix(y_true, y_pred, labels, func):

    # confusion_matrix = np.identity(len(labels))
    confusion_matrix = np.zeros((len(labels),len(labels)))

    for i in range(0,len(y_true)):
        words_true = y_true[i].split(" ")
        words_pred = y_pred[i].split(" ")
        for j in range(0, len(words_true)):
            phonemes_true = func(words_true[j])
            phonemes_pred = func(words_pred[j])
            max_length = min(len(phonemes_true),len(phonemes_pred))
            phonemes_true = phonemes_true[:max_length]
            phonemes_pred = phonemes_pred[:max_length]

            try:
                confusion_matrix = np.add(
                    confusion_matrix,
                    metrics.confusion_matrix(phonemes_true, phonemes_pred, labels=labels)
                )
            except:
                continue

    return confusion_matrix



y_true_path = sys.argv[1]
y_pred_path = sys.argv[2]

with open(y_true_path) as f:
    y_true_r = f.read().splitlines()
with open(y_pred_path) as f:
    y_pred_r = f.read().splitlines()

y_true = []
y_pred = []
for i in range(0,len(y_true_r)):
    if y_true_r[i] in y_true:
        continue
    y_true.append(y_true_r[i])
    y_pred.append(y_pred_r[i])

for k in range(0, len(SCENARIOS)):
    _name = SCENARIOS[k][0]
    _labels = SCENARIOS[k][1]
    confusion_matrix = get_confusion_matrix(y_true,y_pred,_labels,get_phonemes)

    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=_labels, normalize=True,
                          title=_name)

    # plt.show()
    savepath = os.path.join('confusions', _name + '.png')
    print savepath
    plt.savefig(savepath, bbox_inches='tight')


# INTRA-VISEMES
viseme_name = 'Intra-visemes'
viseme_labels = ['V', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
confusion_matrix = get_confusion_matrix(y_true,y_pred,viseme_labels,get_viseme)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=viseme_labels, normalize=True,
                      title=viseme_name)

# plt.show()
savepath = os.path.join('confusions', viseme_name + '.png')
print savepath
plt.savefig(savepath, bbox_inches='tight')