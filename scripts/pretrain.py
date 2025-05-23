import sys
sys.path.append('..')

import argparse
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from src.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from src.data.pretrain_dataset import MoleculeDataset
from src.data.collator import Collator_pretrain
from src.model.get_model import get_pretrain_model
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
from src.loss import DistributedNCELoss

import warnings
warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for pretraining")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train_mode", type=str, required=True, choices=["pretrained", "scratch"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    args = parse_args()
    config = config_dict[args.config]
    print(config)
    torch.backends.cudnn.benchmark = True
    print(local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    val_results, test_results, train_results = [], [], []
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
    train_dataset = MoleculeDataset(root_path=args.data_path)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=config['batch_size']// args.n_devices, num_workers=args.n_threads, worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)

    model = get_pretrain_model(args, train_dataset, local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=20000, tot_updates=200000, lr=config['lr'], end_lr=1e-9, power=1)
    reg_loss_fn = MSELoss(reduction='none')
    clf_loss_fn = BCEWithLogitsLoss(weight=train_dataset._task_pos_weights.to(device),reduction='none')
    sl_loss_fn = CrossEntropyLoss(reduction='none')
    nce_loss_fn = DistributedNCELoss()
    reg_metric, clf_metric = "r2", "rocauc_resp"
    reg_evaluator = Evaluator("reg", reg_metric, train_dataset.d_mds)
    clf_evaluator = Evaluator("clf", clf_metric, train_dataset.d_fps)
    result_tracker = Result_Tracker(reg_metric)
    if local_rank == 0:
        summary_writer = SummaryWriter(f"tensorboard/pretrain-{args.config}")
    else: 
        summary_writer = None
    trainer = Trainer(args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, nce_loss_fn, reg_evaluator, clf_evaluator, result_tracker, summary_writer, device=device,local_rank=local_rank)
    trainer.fit(model, train_loader)
    if local_rank == 0:
        summary_writer.close()
