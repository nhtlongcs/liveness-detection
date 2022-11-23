# split data 95 5 train test
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

labels_path = Path(sys.argv[1])

assert labels_path.exists(), f"Labels file {labels_path} does not exist"
assert labels_path.suffix == ".csv", f"Labels file {labels_path} should be a csv file"

labels_df = pd.read_csv(labels_path)

SEED = 128

MUST_INCLUDE_IN_VALIDATION = [
    42, 131, 10, 780, 941, 1104, 864, 698, 641, 694, 299, 343, 246, 454, 657
]  # same persons but different videos (check eda.ipynb)

MUST_INCLUDE_IN_VALIDATION += [316, 1, 551, 755, 738, 693,
                               693]  # Same videos (check eda.ipynb)
MUST_INCLUDE_IN_VALIDATION += [249, 251, 334, 418, 694, 641,
                               244]  # Exactly same videos (check eda.ipynb)

# MUST_INCLUDE_IN_VALIDATION += [367, 313]  # MISC

NOT_USE_VIDEOS = [674]


# split by video
def split(df, train_size=0.95, key="filename"):
    # remove NOT_USE_VIDEOS
    global NOT_USE_VIDEOS, MUST_INCLUDE_IN_VALIDATION

    NOT_USE_VIDEOS = [f'{i}.mp4' for i in NOT_USE_VIDEOS]
    df = df[~df["filename"].isin(NOT_USE_VIDEOS)]
    train, test = train_test_split(df[key].unique(),
                                   train_size=train_size,
                                   random_state=SEED)
    # MUST_INCLUDE_IN_VALIDATION
    MUST_INCLUDE_IN_VALIDATION = [
        f'{i}.mp4' for i in MUST_INCLUDE_IN_VALIDATION
    ]
    test = list(set(test).union(set(MUST_INCLUDE_IN_VALIDATION)))
    train = list(set(train) - set(MUST_INCLUDE_IN_VALIDATION))
    train_df = df[df[key].isin(train)]
    test_df = df[df[key].isin(test)]
    return train_df, test_df


def sanity_check(df_raw, df_train, df_test):
    assert len(df_raw) - len(NOT_USE_VIDEOS) == len(df_train) + len(df_test)
    assert set(df_test["filename"].unique()).intersection(
        set(df_train["filename"].unique())) == set(
        ), "Train and test sets should not have any common videos"


train_df, test_df = split(labels_df)
sanity_check(labels_df, train_df, test_df)
print(
    f"Train set size: {len(train_df)}, video count: {len(train_df['filename'].unique())}"
)
print(
    f"Test set size: {len(test_df)}, video count: {len(test_df['filename'].unique())}"
)

train_df.to_csv(f"{str(labels_path.parent)}/labels_video_train.csv",
                index=False)
test_df.to_csv(f"{str(labels_path.parent)}/labels_video_test.csv", index=False)
