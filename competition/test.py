from metrics import compute_eer, compute_acc
import pandas as pd
import tabulate
from argparse import ArgumentParser

if __name__ == "__main__":
    # label = [1, 1, 0, 0, 1]
    # prediction = [0.3, 0.1, 0.4, 0.8, 0.9]
    # eer = compute_eer(label, prediction)
    parser = ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()
    gt_df = pd.read_csv(args.gt)
    pred_df = pd.read_csv(args.pred)

    # get intersection of filenames
    filenames = list(
        set(gt_df["filename"]).intersection(set(pred_df["fname"])))
    error_filenames = set(gt_df["filename"]) - set(filenames)
    for filename in error_filenames:
        print("Error: {} not found in prediction file".format(filename))

    gt_df = gt_df[gt_df["filename"].isin(filenames)]
    pred_df = pred_df[pred_df["fname"].isin(filenames)]

    # sort by filename
    gt_df = gt_df.sort_values(by="filename")
    pred_df = pred_df.sort_values(by="fname")
    # print(gt_df["filename"].values)
    # print(pred_df["filename"].values)

    label = gt_df["label"].values.tolist()
    prediction = pred_df["liveness_score"].values.tolist()
    # print(gt_df["label"].values)
    # print(pred_df["label"].values)
    assert len(label) == len(
        prediction), "Error: length of label and prediction are not equal"
    # print number of ground truth labels and predictions as table
    table = [["number of gt sample", "number of predictions"],
             [len(label), len(prediction)]]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    eer = compute_eer(label, prediction)
    acc = compute_acc(label, prediction, threshold=0.5)
    # print('The equal error rate is {:.8f}'.format(eer))
