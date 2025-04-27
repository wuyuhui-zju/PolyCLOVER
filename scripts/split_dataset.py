import argparse
import os
import pandas as pd
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_train", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    num_data = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv").values.shape[0]
    a = np.arange(num_data)

    np.random.seed(args.seed)
    np.random.shuffle(a)
    split_index = int(len(a) * args.n_train)

    split = []
    split.append(a[:split_index])
    split.append(a[split_index:])
    print(split)

    save_dir = f'{args.data_path}/{args.dataset}/splits'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'split.pkl'), 'wb') as f:
        pickle.dump(split, f)
