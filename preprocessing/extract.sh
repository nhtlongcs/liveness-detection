# extract a folder of videos to frames
# Usage: extract.sh <video_folder> <output_folder>
VIDEO_FOLDER=$1
OUTPUT_FOLDER=$2

# create output folder if not exist
mkdir -p $OUTPUT_FOLDER

# extract frames from videos
for video in $VIDEO_FOLDER/*.mp4
do
    echo "Extracting $video"
    filename=$(basename -- "$video")
    filename="${filename%.*}"
    ffmpeg -i $video -qscale:v 2 $OUTPUT_FOLDER/$filename-%05d.jpg
done