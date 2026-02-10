import torch
from src.ds.cmnistbar_data_loader import ColoredBarMNIST

import numpy as np
import pandas as pd

def get_obs_data_table(graph, root, batch_size=400):
    dat_set = ColoredBarMNIST(cg="full-ncm", root=root, env='train', v_only=True)
    dat_loader = torch.utils.data.DataLoader(dat_set, batch_size=batch_size, shuffle=True,
                                                drop_last=False)

    y_tol = []

    for idx, y in enumerate(dat_loader):
        y_tol.append(y)


    y_tol = torch.cat(y_tol, dim=0)

    if graph == "full-ncm":
        dat = {
            'D': torch.argmax(y_tol[:, :10], dim=1, keepdim=True),
            'C': torch.argmax(y_tol[:, 10:12], dim=1, keepdim=True),
            'BC': torch.argmax(y_tol[:, 12:14], dim=1, keepdim=True),
            'BW': torch.argmax(y_tol[:, 14:16], dim=1, keepdim=True),
        }
    elif graph == "cls-digit":
        dat = {
            'X': torch.argmax(y_tol[:, :10], dim=1, keepdim=True),
            'B': torch.argmax(y_tol[:, 10:12], dim=1, keepdim=True),
            'Z': torch.argmax(y_tol[:, 12:14], dim=1, keepdim=True),
        }
    elif graph == "cls-color":
        dat = {
            'B': torch.argmax(y_tol[:, :10], dim=1, keepdim=True),
            'X': torch.argmax(y_tol[:, 10:12], dim=1, keepdim=True),
        }

    cols = dict()
    for v in sorted(dat):
        result = dat[v].detach().cpu().numpy()
        for i in range(result.shape[1]):
            cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

    df = pd.DataFrame(cols)
    grouped = (df.groupby(list(df.columns))
                .apply(lambda x: len(x) / len(df))
                .rename('P(V)').reset_index()
                [[*df.columns, 'P(V)']])

    if graph == "full-ncm":
        ordered_cols = ['D0', 'C0', 'BC0', 'BW0', 'P(V)']
    elif graph == "cls-digit":
        ordered_cols = ['X0', 'B0', 'Z0', 'P(V)']
    elif graph == "cls-color":
        ordered_cols = ['X0', 'B0', 'P(V)']

    grouped = grouped[ordered_cols]
    grouped = grouped.sort_values(ordered_cols[:-1]).reset_index(drop=True)
    return grouped