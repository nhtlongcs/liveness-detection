# split data 95 5 train test
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

labels_df = pd.read_csv("data/train/labels_keyframes.csv")
labels_df.sort_values(by=["video_id", "frame_id"], inplace=True)
SEED = 128
# split by video
def split(df, train_size=0.95,key="video_id"):
    train, test = train_test_split(df[key].unique(), train_size=train_size, random_state=SEED)
    train_df = df[df[key].isin(train)]
    test_df = df[df[key].isin(test)]
    return train_df, test_df

def sanity_check(df_raw, df_train, df_test):
    assert len(df_raw) == len(df_train) + len(df_test)
    assert set(df_test["video_id"].unique()).intersection(set(df_train["video_id"].unique())) == set(), "Train and test sets should not have any common videos"

train_df, test_df = split(labels_df)
sanity_check(labels_df, train_df, test_df)
print(f"Train set size: {len(train_df)}, video count: {len(train_df['video_id'].unique())}")
print(f"Test set size: {len(test_df)}, video count: {len(test_df['video_id'].unique())}")

train_df.to_csv("data/train/labels_keyframes_train.csv", index=False)
test_df.to_csv("data/train/labels_keyframes_test.csv", index=False)