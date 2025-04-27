import os
import numpy as np
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from src.data.collator import Collator_AL
from src.data.inference_dataset import InferenceDataset
from src.model.light import LiGhTPredictor as LiGhT, TripletEmbeddingCopoly
from src.model_config import config_dict
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


def get_triplet_emb_layer(d_g_feats, d_fp_feats, d_md_feats, device):
    triplet_emb_layer = TripletEmbeddingCopoly(d_g_feats, d_fp_feats, d_md_feats)
    triplet_emb_layer.apply(lambda module: init_params(module))
    return triplet_emb_layer.to(device)


def get_global_node_emb(d_g_feats, device):
    global_node_emb_layer = nn.Embedding(1, d_g_feats)
    global_node_emb_layer.apply(lambda module: init_params(module))
    return global_node_emb_layer.to(device)


class EnsembleRegressor:
    def __init__(self, args):
        config = config_dict[args.config]
        self.config = config
        self.args = args

        self.preference = config["preference"]
        self.device = args.device
        self.train_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, split='train')
        self.model_fns = os.listdir(args.model_path)

    def forward_with_uncertainty(self, dataloader):
        s_ensemble = []
        task_1_ensemble = []
        task_2_ensemble = []
        for i in range(self.config["num_ensemble"]):
            print(f"Base learner: {i}......")
            model = self.load_model(self.config, self.args, i)
            predictions = self.predict(model, dataloader)
            s = self.preference * predictions[:, 0] + (1 - self.preference) * predictions[:, 1]
            s_ensemble.append(s.tolist())
            task_1_ensemble.append(predictions[:, 0].tolist())
            task_2_ensemble.append(predictions[:, 1].tolist())
        s_ensemble = np.array(s_ensemble)
        task_1_ensemble = np.array(task_1_ensemble)
        task_2_ensemble = np.array(task_2_ensemble)

        return s_ensemble.mean(axis=0), s_ensemble.std(axis=0), task_1_ensemble.mean(axis=0), task_2_ensemble.mean(axis=0)

    def calculate_latent_embeddings(self, dataloader):
        model = self.load_model(self.config, self.args, i=0)
        embs_all = []

        for batched_data in dataloader:
            (smiles_1, smiles_2, smiles_3, g, fp_1, md_1, fp_2, md_2, fp_3, md_3, ratio) = batched_data
            fp_1 = fp_1.to(self.device)
            md_1 = md_1.to(self.device)
            fp_2 = fp_2.to(self.device)
            md_2 = md_2.to(self.device)
            fp_3 = fp_3.to(self.device)
            md_3 = md_3.to(self.device)
            ratio = ratio.to(self.device)
            g = g.to(self.device)
            embs = model.generate_fps(g, fp_1, md_1, fp_2, md_2, fp_3, md_3, ratio)
            embs_all.append(embs.detach().cpu())

        return torch.cat(embs_all).numpy()

    def load_model(self, config, args, i):
        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=self.train_dataset.d_fps,
            d_md_feats=self.train_dataset.d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        ).to(self.device)
        model.triplet_emb = get_triplet_emb_layer(d_g_feats=config['d_g_feats'], d_fp_feats=self.train_dataset.d_fps, d_md_feats=self.train_dataset.d_mds, device=self.device)
        model.triplet_emb.embedding = nn.Embedding(10, config['d_g_feats']).to(self.device)
        model.triplet_emb.global_node_emb = get_global_node_emb(d_g_feats=config['d_g_feats'], device=self.device)
        model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4, n_tasks=2, n_layers=2, predictor_drop=0, device=self.device, d_hidden_feats=256)
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        del model.cl_projector
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}/{self.model_fns[i]}').items()})
        model.eval()
        return model

    def predict(self, model, dataloader):
        model.eval()
        predictions_all = []

        for batched_data in dataloader:
            (smiles_1, smiles_2, smiles_3, g, fp_1, md_1, fp_2, md_2, fp_3, md_3, ratio) = batched_data
            fp_1 = fp_1.to(self.device)
            md_1 = md_1.to(self.device)
            fp_2 = fp_2.to(self.device)
            md_2 = md_2.to(self.device)
            fp_3 = fp_3.to(self.device)
            md_3 = md_3.to(self.device)
            ratio = ratio.to(self.device)
            g = g.to(self.device)
            predictions = model.forward_tune(g, fp_1, md_1, fp_2, md_2, fp_3, md_3, ratio)
            predictions_all.append(predictions.detach().cpu())

        predictions = torch.cat(predictions_all).numpy()

        train_dataset_std = self.train_dataset.std.numpy()
        train_dataset_mean = self.train_dataset.mean.numpy()
        for i in range(predictions.shape[1]):
            predictions[:, i] = predictions[:, i] * train_dataset_std[i] + train_dataset_mean[i]

        return predictions


class UCB:
    def __init__(self, kappa):
        self.kappa = kappa

    def __call__(self, mean, std):
        return mean - self.kappa * std


class Proxy:
    def __init__(self, args, model, acq_fn, dim_reducer, clustering):
        self.args = args
        self.config = config_dict[args.config]
        self.model = model
        self.acq_fn = acq_fn
        self.dim_reducer = dim_reducer
        self.clustering = clustering

    def __call__(self, df_new_dataset):
        new_dataset_loader = self.process_dataset(df_new_dataset)
        mean, std, task_1_mean, task_2_mean = self.model.forward_with_uncertainty(new_dataset_loader)
        acq = self.acq_fn(mean, std)

        df_top_dataset = self.get_top_dataset(df_new_dataset, acq, mean, std, task_1_mean, task_2_mean)
        top_loader = self.process_dataset(df_top_dataset)
        embs = self.model.calculate_latent_embeddings(top_loader)

        # dim reduction
        print("Clustering...")
        scaler = StandardScaler()
        scaler.fit(embs)
        embs = scaler.transform(embs)
        embs = self.dim_reducer.fit_transform(embs)

        # clustering
        labels = self.clustering.fit_predict(embs)  # np.array [num,]
        df_result = self.get_candidate(df_top_dataset, labels)
        return df_result

    def process_dataset(self, df) -> DataLoader:
        dataset = InferenceDataset(df, self.args.data_path)
        collator = Collator_AL()
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=self.args.n_threads,
                            drop_last=False, collate_fn=collator)
        return loader

    def get_top_dataset(self, df_dataset, acq, mean, std, task_1_mean, task_2_mean) -> pd.DataFrame:
        df_dataset["acq"] = acq
        df_dataset["s_mean"] = mean
        df_dataset["s_std"] = std
        df_dataset["task1"] = task_1_mean
        df_dataset["task2"] = task_2_mean
        thresh = int(self.config["top_ratio"] * len(df_dataset))
        print(f"top-{thresh} was selected")
        df_top_dataset = df_dataset.sort_values('acq').iloc[:thresh, :]

        return df_top_dataset

    def get_candidate(self, df_top_dataset, labels: np.array) -> pd.DataFrame:
        df_top_dataset["labels"] = labels
        df_top_dataset = df_top_dataset.reset_index(drop=True)

        # n sample per class
        result = df_top_dataset.groupby('labels', group_keys=False).apply(lambda x: x.nsmallest(self.config["num_samples_per_class"], 'acq'))

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for mobo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)

    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # load tested ID
    config = config_dict[args.config]
    df_database = pd.read_csv(os.path.join(args.data_path, "database/database.csv"))
    tested_ID = pd.read_csv(os.path.join(args.data_path, f"{args.dataset}/{args.dataset}.csv"))["ID"].values.tolist()
    df_new_dataset = df_database.drop(tested_ID, axis=0).reset_index(drop=True)

    ensemble_regressor = EnsembleRegressor(args)
    ucb = UCB(kappa=config["kappa"])
    dim_reducer = TSNE(n_components=2, random_state=44)
    clustering = KMeans(n_clusters=config["n_clusters"], random_state=44)

    proxy = Proxy(args, ensemble_regressor, ucb, dim_reducer, clustering)
    df_result = proxy(df_new_dataset)

    df_database_raw = pd.read_csv("../datasets/database/database_raw.csv")
    df_result_merge = pd.merge(df_result, df_database_raw, on='ID', how='left')

    save_dir = "../results"
    os.makedirs(save_dir, exist_ok=True)
    df_result_merge.to_csv(f"../results/{args.dataset}_candidate.csv", index=False)
