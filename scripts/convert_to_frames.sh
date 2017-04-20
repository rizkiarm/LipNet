LIST=$(ls 0_Data/$1/*)

for VID in $LIST
do
    NEW="${VID/0_Data/1_DataFrames}" # Target
    NEW="${NEW/.mpg/}"
    mkdir -p $NEW
    ffmpeg -i $VID -f image2 $NEW/image%03d.png
done
