# create a folder to store extracted images
import os
folder = 'test'
os.mkdir(folder)
# use opencv to do the job
import cv2
import sys
print(cv2.__version__)  # my version is 3.1.0

DATASET_VIDEO_PATH = sys.argv[1]

print("Will be saved to", DATASET_VIDEO_PATH)

vidcap = cv2.VideoCapture('swwv9a_mouth.mp4')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))
