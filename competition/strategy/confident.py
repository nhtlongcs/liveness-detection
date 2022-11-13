import pandas as pd 
import numpy as np
from argparse import ArgumentParser

def confident_strategy(pred, t=0.87):
   pred = np.array(pred)
   size = len(pred)
   fakes = np.count_nonzero(pred > t)
   if fakes > size // 3 and fakes > 11:
       return np.mean(pred[pred > t])
   elif np.count_nonzero(pred < 0.2) > 0.6 * size:
       return np.mean(pred[pred < 0.2])
   else:
       return np.mean(pred)
# write arg parser
parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

keyframe_predict = pd.read_csv(args.input)
video_ids = keyframe_predict["video_id"].unique()
result_video_ids = []
result_labels = []

for video_id in video_ids:
    video_df = keyframe_predict[keyframe_predict["video_id"] == video_id]
    video_df = video_df.sort_values(by="frame_id")
    video_df = video_df.reset_index(drop=True)
    lbl = confident_strategy(video_df["prob"].values)
    result_video_ids.append(video_id)
    result_labels.append(lbl)
result_df = pd.DataFrame({"video_id": result_video_ids, "label": result_labels})
result_df.sort_values(by="video_id", inplace=True)
result_df['fname'] = result_df['video_id'].apply(lambda x: str(x) + '.mp4')
result_df['liveness_score'] = result_df['label']
result_df = result_df[['fname', 'liveness_score']]
print(len(result_df))
result_df.to_csv(args.output, index=False,  float_format='%.10f')