import torch
from torch import nn

import sys
sys.path.append('..')
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.model.light import LiGhTPredictor as LiGhT, TripletEmbeddingCopoly
from src.model_config import config_dict


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
        for _ in range(n_layers-2):
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


def get_pretrain_model(args, train_dataset, local_rank):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    device = torch.device('cuda', local_rank)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_cl_feats=config['d_cl_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=config['input_drop'],
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to(device)
    del model.cl_projector
    model.triplet_emb = get_triplet_emb_layer(d_g_feats=config['d_g_feats'], d_fp_feats=train_dataset.d_fps, d_md_feats=train_dataset.d_mds, device=device)
    if args.train_mode == "pretrain":
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()}, strict=False)
    model.triplet_emb.embedding = nn.Embedding(10, config['d_g_feats']).to(device)
    model.triplet_emb.global_node_emb = get_global_node_emb(d_g_feats=config['d_g_feats'], device=device)
    model.cl_projector = nn.Sequential(nn.Linear(config['d_g_feats'] * 4, config['d_g_feats']), nn.GELU(),
                                       nn.Linear(config['d_g_feats'], config['d_cl_feats'])).to(device)

    return model


def get_finetune_model(args, train_dataset):
    config = config_dict[args.config]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)

    model.triplet_emb = get_triplet_emb_layer(d_g_feats=config['d_g_feats'], d_fp_feats=train_dataset.d_fps, d_md_feats=train_dataset.d_mds, device=device)
    model.triplet_emb.embedding = nn.Embedding(10, config['d_g_feats']).to(device)
    model.triplet_emb.global_node_emb = get_global_node_emb(d_g_feats=config['d_g_feats'], device=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()})  # two-stage pretrain
    model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=args.dropout, device=device, d_hidden_feats=256)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    del model.cl_projector
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))

    return model
