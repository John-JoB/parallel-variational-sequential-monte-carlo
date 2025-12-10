import torch
import numpy as np
import einops
import pandas as pd
from pathlib import Path

sub_path_length = 120

def get_lags(spx, t):
    return np.lib.stride_tricks.as_strided(spx, shape=(spx.shape[0] - t + 1, t), strides=(spx.strides[0], spx.strides[0]))

if __name__ == "__main__":
    data_path = Path("./experiments/SPX/data/")
    raw_data = np.load(data_path / "raw.npy")
    spx = raw_data[:, 0]
    vix = raw_data[:, 1]
    spx_lags = get_lags(spx, sub_path_length).copy()
    spx_lags = spx_lags[1:] / spx_lags[:-1, 0:1]
    spx_lags = spx_lags.flatten()
    vix = vix[:spx.shape[0] - sub_path_length]
    metadata_series_id = np.arange(vix.shape[0])
    series_id = metadata_series_id.repeat(sub_path_length)
    df = pd.DataFrame(spx_lags, columns=[f'observation_1'])
    df['series_id'] = series_id
    df.to_csv(data_path / "data.csv", index=False)
    metadata = pd.DataFrame(vix, columns=[f'metadata_1'])
    metadata["series_id"] = metadata_series_id
    metadata.to_csv(data_path / "metadata.csv", index=False)
