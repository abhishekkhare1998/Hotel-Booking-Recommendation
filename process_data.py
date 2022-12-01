import os
import pandas as pd

def downsample_data(input_df, sampling_factor=2):
    downsampled_df = input_df.iloc[::sampling_factor, :]
    return downsampled_df


def main_function():
    os.chdir(os.path.realpath(__file__).rsplit("\\", 1)[0])
    orig_dset = pd.read_csv("expedia-hotel-recommendations/train_small.csv")
    training_Set_df = orig_dset.dropna()
    sampling_factor = 32
    downsampled_df = downsample_data(training_Set_df, sampling_factor=sampling_factor)
    downsampled_df.to_csv("expedia-hotel-recommendations/downsampled_output_{}.csv".format(sampling_factor))

    return


if __name__=='__main__':
    main_function()