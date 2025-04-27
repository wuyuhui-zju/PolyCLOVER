import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

import sys
sys.path.append('..')
from src.data.eval_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.eval_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.utils import set_random_seed
from src.model.get_model import get_finetune_model
import warnings
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for evaluating test performance")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="../datasets/")
    parser.add_argument("--metric", type=str, default="rmse")

    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_suffix", type=str)
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
    test_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    model = get_finetune_model(args, train_dataset)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs*len(train_dataset)//32//10, tot_updates=args.n_epochs*len(train_dataset)//32,lr=args.lr, end_lr=1e-9,power=1)
    loss_fn = MSELoss(reduction='none')
    evaluator = Evaluator(args.dataset, "rmse", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_rmse = Evaluator(args.dataset, "rmse", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_top5 = Evaluator(args.dataset, "rmse_top5", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_top10 = Evaluator(args.dataset, "rmse_top10", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_r2 = Evaluator(args.dataset, "r2", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_r = Evaluator(args.dataset, "r", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator_mae = Evaluator(args.dataset, "mae", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator = [final_evaluator_rmse, final_evaluator_top5, final_evaluator_top10, final_evaluator_r2, final_evaluator_r, final_evaluator_mae]
    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    print(f"mean: {train_dataset.mean.numpy()}\tstd: {train_dataset.std.numpy()}")
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator, result_tracker, summary_writer, device=device,model_name='Base', label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
    best_train, best_val, best_test, final_metrics = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")
    print(f"test rmse: {final_metrics[0]:.3f}")
    print(f"test top5rmse: {final_metrics[1]:.3f}")
    print(f"test top10rmse: {final_metrics[2]:.3f}")
    print(f"test r2: {final_metrics[3][0]:.3f}")
    print(f"test r: {final_metrics[4]:.3f}")
    print(f"test mae: {final_metrics[5]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    finetune(args)
