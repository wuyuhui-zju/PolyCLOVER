config_dict = {
    'base': {
        'd_node_feats': 138,
        'd_edge_feats': 14,
        'd_g_feats': 768,
        'd_cl_feats': 256,
        'd_hpath_ratio': 12,
        'n_mol_layers': 12,
        'path_length': 5,
        'n_heads': 12,
        'n_ffn_dense_layers': 2,
        'input_drop': 0.0,
        'attn_drop': 0.1,
        'feat_drop': 0.1,
        'batch_size': 96,
        'lr': 2e-04,
        'weight_decay': 1e-6,
        'candi_rate': 0.5,
        'fp_disturb_rate': 0.5,
        'md_disturb_rate': 0.5
    },

    'al': {
        'd_node_feats': 138,
        'd_edge_feats': 14,
        'd_g_feats': 768,
        'd_cl_feats': 256,
        'd_hpath_ratio': 12,
        'n_mol_layers': 12,
        'path_length': 5,
        'n_heads': 12,
        'n_ffn_dense_layers': 2,
        "top_ratio": 0.1,
        "n_clusters": 20,
        "num_samples_per_class": 3,
        "preference": 0.7,
        "kappa": 2,
        "num_ensemble": 20
    }
}