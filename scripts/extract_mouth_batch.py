''' 
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.

Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]

    pattern: *.avi, *.mpg, etc 

Example:
    python scripts/extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat

    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''

from lipnet.lipreading.videos import Video
import os, fnmatch, sys, errno  
from skimage import io

SOURCE_PATH = sys.argv[1]
SOURCE_EXTS = sys.argv[2]
TARGET_PATH = sys.argv[3]

FACE_PREDICTOR_PATH = sys.argv[4]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print ("Processing: {}".format(filepath))
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)

    filepath_wo_ext = os.path.splitext(filepath)[0]
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    mkdir_p(target_dir)

    i = 0
    for frame in video.mouth:
    	io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
    	i += 1
