import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse as sp
import argparse

import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from mordred import Calculator, descriptors

import sys
sys.path.append("..")
from src.data.featurizer import smiles_to_graph_tune
from src.utils import generate_oligomer_smiles
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


def preprocess_dataset(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}.pkl"
    smiless_1 = df["B"].values.tolist()
    smiless_2 = df["C"].values.tolist()
    smiless_3 = df["D"].values.tolist()

    monomer_list = []
    product_list_1 = []
    product_list_2 = []
    product_list_3 = []
    for smiles_1, smiles_2, smiles_3 in zip(smiless_1, smiless_2, smiless_3):
        try:
            product_smiles_1 = generate_oligomer_smiles(num_repeat_units=3, smiles=smiles_1)
            product_smiles_2 = generate_oligomer_smiles(num_repeat_units=3, smiles=smiles_2)
            product_smiles_3 = generate_oligomer_smiles(num_repeat_units=3, smiles=smiles_3)

            monomer_list.append([smiles_1, smiles_2, smiles_3])
            product_list_1.append(product_smiles_1)
            product_list_2.append(product_smiles_2)
            product_list_3.append(product_smiles_3)
        except:
            print(smiles_1, smiles_2, smiles_3)
            continue
    task_names = ['value1', 'value2']
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  monomer_list,
                  max_length=5,
                  n_knowledge_nodes=3, n_global_nodes=1,
                  n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print(f'saving graphs len(graphs)={len(valid_ids)}')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    for i, product_list in enumerate([product_list_1, product_list_2, product_list_3], 1):
        print(f'extracting fingerprints: {i}')
        fp_list = []
        for smiles in product_list:
            mol = Chem.MolFromSmiles(smiles)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
            fp_list.append(list(map(int, list(maccs_fp + ec_fp))))
        fp_list = np.array(fp_list, dtype=np.float32)
        print(f"fp shape: {fp_list.shape}")
        fp_sp_mat = sp.csc_matrix(fp_list)
        print('saving fingerprints')
        sp.save_npz(f"{args.data_path}/{args.dataset}/maccs_ec_fp_{i}.npz", fp_sp_mat)

        print(f'extracting molecular descriptors: {i}')
        des_list = []
        for smiles in tqdm(product_list, total=len(product_list)):
            calc = Calculator(descriptors, ignore_3D=True)
            mol = Chem.MolFromSmiles(smiles)
            des = np.array(list(calc(mol).values()), dtype=np.float32)
            des_list.append(des)
        des = np.array(des_list)
        des = np.where(np.isnan(des), 0, des)
        des = np.where(des > 10**12, 10**12, des)

        with open(f"{args.data_path}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        des_norm = scaler.transform(des)

        print(f"des shape: {des.shape}")
        np.savez_compressed(f"{args.data_path}/{args.dataset}/molecular_descriptors_{i}.npz", md=des_norm)

    print('extracting ratio des')
    ratio_des = df.iloc[:, 4:-2].values
    np.savez_compressed(f"{args.data_path}/{args.dataset}/ratio.npz", ratio=ratio_des)


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)
