import os
import pandas as pd
import numpy as np
import scipy.sparse as sps

from dgl.data.utils import load_graphs
import torch
from torch.utils.data import Dataset


SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}


class InferenceDataset(Dataset):
    def __init__(self, df_new_dataset, root_path):
        dataset_path = os.path.join(root_path, f"database/database.csv")
        self.cache_path = os.path.join(root_path, f"database/database.pkl")
        fp_path_1 = os.path.join(root_path, f"database/maccs_ec_fp_1.npz")
        md_path_1 = os.path.join(root_path, f"database/molecular_descriptors_1.npz")
        fp_path_2 = os.path.join(root_path, f"database/maccs_ec_fp_2.npz")
        md_path_2 = os.path.join(root_path, f"database/molecular_descriptors_2.npz")
        fp_path_3 = os.path.join(root_path, f"database/maccs_ec_fp_3.npz")
        md_path_3 = os.path.join(root_path, f"database/molecular_descriptors_3.npz")
        ratio_path = os.path.join(root_path, f"database/ratio.npz")
        # Load Data
        df = pd.read_csv(dataset_path)

        use_idxs = df_new_dataset["ID"].values

        fps_1 = torch.from_numpy(sps.load_npz(fp_path_1).todense().astype(np.float32))
        mds_1 = np.load(md_path_1)['md'].astype(np.float32)
        mds_1 = torch.from_numpy(np.where(np.isnan(mds_1), 0, mds_1))

        fps_2 = torch.from_numpy(sps.load_npz(fp_path_2).todense().astype(np.float32))
        mds_2 = np.load(md_path_2)['md'].astype(np.float32)
        mds_2 = torch.from_numpy(np.where(np.isnan(mds_2), 0, mds_2))

        fps_3 = torch.from_numpy(sps.load_npz(fp_path_3).todense().astype(np.float32))
        mds_3 = np.load(md_path_3)['md'].astype(np.float32)
        mds_3 = torch.from_numpy(np.where(np.isnan(mds_3), 0, mds_3))

        ratio = np.load(ratio_path)['ratio'].astype(np.float32)
        ratio = torch.from_numpy(ratio)*10

        self.df, self.fps_1, self.mds_1, self.fps_2, self.mds_2, self.fps_3, self.mds_3 = \
            df.iloc[use_idxs], fps_1[use_idxs], mds_1[use_idxs], fps_2[use_idxs], mds_2[use_idxs], fps_3[use_idxs], \
            mds_3[use_idxs]
        self.ratio = ratio[use_idxs]

        self.smiless_1 = self.df['B'].tolist()
        self.smiless_2 = self.df['C'].tolist()
        self.smiless_3 = self.df['D'].tolist()
        self.use_idxs = use_idxs

        self._pre_process()
        self.d_fps = self.fps_1.shape[1]
        self.d_mds = self.mds_1.shape[1]

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
        self.fps_1, self.mds_1 = self.fps_1, self.mds_1
        self.fps_2, self.mds_2 = self.fps_2, self.mds_2
        self.fps_3, self.mds_3 = self.fps_3, self.mds_3

    def __len__(self):
        return len(self.smiless_1)

    def __getitem__(self, idx):
        return self.smiless_1[idx], self.smiless_2[idx], self.smiless_3[idx], self.graphs[idx], self.fps_1[idx], \
        self.mds_1[idx], self.fps_2[idx], self.mds_2[idx], self.fps_3[idx], self.mds_3[idx], self.ratio[idx]


if __name__ == "__main__":
    pass