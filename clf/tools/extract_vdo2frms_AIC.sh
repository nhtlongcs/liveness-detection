DATADIR=$1
OUTDIR=$2
TRAINDIR=$DATADIR/train/
VALIDDIR=$DATADIR/validation/

for d in $TRAINDIR/* ; do
    camera_name=$(basename $d)
    for vdo in $d/* ; do
        vdo_name=$(basename $vdo)
        echo "Processing [TRAIN][$camera_name][$vdo_name]"
        outdir=$OUTDIR/train/$camera_name/$vdo_name/img1
        mkdir -p $outdir
        ffmpeg -i $vdo/vdo.avi -loglevel error $outdir/%06d.jpg 
    done
done

for d in $VALIDDIR/* ; do
    camera_name=$(basename $d)
    for vdo in $d/* ; do
        vdo_name=$(basename $vdo)
        echo "Processing [VALID][$camera_name][$vdo_name]"
        outdir=$OUTDIR/validation/$camera_name/$vdo_name/img1
        mkdir -p $outdir
        ffmpeg -i $vdo/vdo.avi -loglevel error $outdir/%06d.jpg 
    done
done