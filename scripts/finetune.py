import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import random

import sys
sys.path.append('..')
from src.utils import set_random_seed
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model.get_model import get_finetune_model

import warnings
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training base learner")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="../datasets/")
    parser.add_argument("--metric", type=str, default="rmse")

    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--ensemble_idx", type=int, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args


def finetune(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    collator = Collator_tune()
    train_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, split='train')
    val_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, split='val')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    model = get_finetune_model(args, train_dataset)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs*len(train_dataset)//32//10, tot_updates=args.n_epochs*len(train_dataset)//32,lr=args.lr, end_lr=1e-9,power=1)
    loss_fn = MSELoss(reduction='none')
    evaluator = Evaluator(args.dataset, "rmse", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator = Evaluator(args.dataset, "r2", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator, result_tracker, summary_writer, device=device,model_name='Base', label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
    best_train, best_val, train_r2, test_r2 = trainer.fit(model, train_loader, val_loader)
    print(f"train: {best_train:.3f}, val: {best_val:.3f}")
    print("----------------------")
    print("Task 1:")
    print(f"train r2: {train_r2[0]:.3f}, test r2: {test_r2[0]:.3f}")
    print("----------------------")
    print("Task 2:")
    print(f"train r2: {train_r2[1]:.3f}, test r2: {test_r2[1]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    finetune(args)
