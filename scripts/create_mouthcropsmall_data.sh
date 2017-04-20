LIST=$(ls -d 1_DataFrames/$1/*/)

for VID in $LIST
do
    NEW="${VID/1_DataFrames/2_DataMouthCropSmall}" # Target
    mkdir -p $NEW
    python face_landmark_detection.py shape_predictor_68_face_landmarks.dat $VID $NEW
done
