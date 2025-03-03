import os
import sys
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))
sys.path.append('/home/yuhaowang/project/FMBC')
import time
import wandb
os.environ["WANDB_API_KEY"] = '6ebb1c769075243eb73a32c4f9f7011ddd41f20a'

import torch
import numpy as np
import torch.utils.tensorboard as tensorboard
from gigapath.classification_head import get_model
from metrics import calculate_metrics_with_task_cfg
from finetune_utils import (
    get_optimizer, get_loss_function, Monitor_Score, get_records_array,
    log_writer, adjust_learning_rate, release_nested_dict,
    initiate_mil_model, initiate_linear_model
)
from fewshot_algorithms import FewShot, SimpleShot, NearestNeighbors

def train(dataloader, fold, args):
    train_loader, val_loader, test_loader = dataloader
    writer_dir = os.path.join(args.save_dir, f'fold_{fold}', 'tensorboard')
    os.makedirs(writer_dir, exist_ok=True)

    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    if "wandb" in args.report_to:
        wandb.init(
            project=args.task,
            name=args.pretrain_model + '_fold_' + str(fold) + '_' + str(args.tuning_method),
            id='fold_' + str(fold) + '_' + str(args.pretrain_model) + '_' + str(args.tuning_method),
            config=vars(args),
            settings=wandb.Settings(init_timeout=120)
        )
        writer = wandb
    
    if args.pretrain_model_type == 'patch_level':
        model = initiate_linear_model(args) if args.tuning_method == 'LR' else initiate_mil_model(args)
    else:
        model = get_model(**vars(args)) if args.pretrain_model == 'FMBC' else initiate_linear_model(args)
    
    model = model.to(args.device)
    optimizer = get_optimizer(args, model)
    loss_fn = get_loss_function(args.task_config)
    monitor = Monitor_Score()
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Select algorithm
    algorithm = None
    if args.algorithm == 'fewshot':
        algorithm = FewShot()
    elif args.algorithm == 'simpleshot':
        algorithm = SimpleShot()
    elif args.algorithm == '20nn':
        algorithm = NearestNeighbors(k=20)
    
    for epoch in range(args.epochs):
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args, algorithm)
        if val_loader:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, epoch, args, algorithm)
            log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
            log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
            log_writer(log_dict, epoch, args.report_to, writer)
            scores = val_records['bacc'] if args.task_config.get('setting') == 'multi_class' else val_records['macro_auroc']
            monitor(scores, model, ckpt_name=os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))

    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")))
    test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, epoch, args, algorithm)
    log_writer({'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}, fold, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None
    return val_records, test_records

def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args, algorithm):
    model.train()
    records = get_records_array(len(train_loader), args.n_classes)
    
    for batch_idx, batch in enumerate(train_loader):
        images, img_coords, pad_mask, label = batch['imgs'].to(args.device), batch['coords'].to(args.device), batch['pad_mask'].to(args.device), batch['labels'].to(args.device)
        
        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            logits = model(images, img_coords, pad_mask)
            if algorithm:
                logits = algorithm.apply(logits)
            loss = loss_fn(logits, label.squeeze(-1).long()) / args.gc
        
        if fp16_scaler is None:
            loss.backward()
            if (batch_idx + 1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            fp16_scaler.scale(loss).backward()
            if (batch_idx + 1) % args.gc == 0:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer.zero_grad()
        records['loss'] += loss.item() * args.gc
    
    records['loss'] /= len(train_loader)
    return records

def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, algorithm):
    model.eval()
    records = get_records_array(len(loader), args.n_classes)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, img_coords, pad_mask, label = batch['imgs'].to(args.device), batch['coords'].to(args.device), batch['pad_mask'].to(args.device), batch['labels'].to(args.device)
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits = model(images, img_coords, pad_mask)
                if algorithm:
                    logits = algorithm.apply(logits)
                loss = loss_fn(logits, label.squeeze(-1))
            records['loss'] += loss.item()
    
    records['loss'] /= len(loader)
    records.update(release_nested_dict(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config)))
    return records
