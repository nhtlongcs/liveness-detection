import pandas as pd 
def keyframe_df2video_df(keyframe_df):
    keyframe_df = keyframe_df[['video_id', 'label']]
    # only keep 1 row per video
    keyframe_df = keyframe_df.drop_duplicates(subset='video_id', keep='first')
    keyframe_df.sort_values(by='video_id', inplace=True)
    keyframe_df['filename'] = keyframe_df.video_id.apply(lambda x: str(x) + '.mp4')
    keyframe_df = keyframe_df[['filename', 'label']]
    return keyframe_df

trainKeyframe_df = pd.read_csv('data/train/labels_keyframes_train.csv')
testKeyframe_df = pd.read_csv('data/train/labels_keyframes_test.csv')
video_df = pd.read_csv('data/train/labels_video.csv')
train_video_df = keyframe_df2video_df(trainKeyframe_df)
test_video_df = keyframe_df2video_df(testKeyframe_df)
assert len(video_df) == len(train_video_df) + len(test_video_df)
train_video_df.to_csv('data/train/labels_video_train.csv', index=False)
test_video_df.to_csv('data/train/labels_video_test.csv', index=False)
