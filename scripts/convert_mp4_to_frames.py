# create a folder to store extracted images
import os
# use opencv to do the job
import cv2
import sys, fnmatch, errno
print(cv2.__version__)  # my version is 3.1.0

SOURCE_PATH = sys.argv[1]
SOURCE_EXTS = sys.argv[2]
TARGET_PATH = sys.argv[3]


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


print("Will be saved to", TARGET_PATH)

for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print("Processing:", filepath)
    vidcap = cv2.VideoCapture(filepath)
    count = 0
    filepath_wo_ext = os.path.splitext(filepath)[0]
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    mkdir_p(target_dir)
    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(target_dir,"frame{:d}.png".format(count)), image)     # save frame as JPEG file
        count += 1
    print("{} images are extacted in {}.".format(count,target_dir))
