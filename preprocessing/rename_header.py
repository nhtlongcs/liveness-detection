# rename column names in the csv file

import pandas as pd
import sys


def rename_header(input_file, output_file):
    df = pd.read_csv(input_file)
    df.columns = ['filename', 'label']
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    rename_header(sys.argv[1], sys.argv[2])
