import os
import json
import utils
import torch
import logging
import timeit
import numpy as np
from evaluate import evaluate
import models.data_loader as data_loader
import torchvision.transforms.functional as TF
from models.models import *
from apex import amp
import apex
from torch.utils.tensorboard import SummaryWriter
from parallel import DataParallelModel, DataParallelCriterion

DAMAGE = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def train_epoch(model, loss_fn, optimizer, dataloader, scheduler, device):
    model.train()
    loss_avg = utils.RunningAverage()
    for X, y_true in dataloader:
        X, y_true = X.to(device), y_true.to(device)
        y_pred = model(X)
                    
        if DAMAGE:
            loss = 0.0
            for weight, i in zip([0.05, 0.2, 0.8, 0.7, 0.4], range(5)):
                loss += weight * loss_fn(y_pred[:, i], y_true[:, :, :, i].float())
        else:
            loss = loss_fn(y_pred.squeeze(), y_true.float())

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        loss_avg.update(loss.item())
    if amp._amp_state.loss_scalers[0]._unskipped != 0:
        scheduler.step()
    return loss_avg()

if __name__ == '__main__':
    utils.set_seed()
    device, params, output_dir = utils.prepare_env()
    with open(os.path.join(output_dir, 'params.json'), 'w') as outfile:
        json.dump(params, outfile)

    writer = SummaryWriter(output_dir)
    logging.info("Fetching dataloaders...")
    train_loader = data_loader.fetch_dataloader(params['train_data'], params['batch_size'], 'Training', DAMAGE)
    val_loader = data_loader.fetch_dataloader(params['val_data'], params['batch_size'], 'Validation', DAMAGE)

    logging.info("Buidling model...")
    if DAMAGE:
        model = Res101_Unet_Double().to(device)
    else:
        model = Res101_Unet_Loc().to(device)
        
    if params['load_model'] is not None:
        logging.info("Loading pretrained weights...")
        model = utils.load_model(model, params['load_model'])
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-6)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    model = torch.nn.DataParallel(model)
#     model = torch.nn.parallel.DistributedDataParallel(model)
    logging.info("Finished building...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    n_epochs = params['epochs']
    best_val_loss, best_f1 = float('Inf'), 0.0
    logging.info("Training...")
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, loss_fn, optimizer, train_loader, scheduler, device)
        if DAMAGE:
            torch.cuda.empty_cache()
        f1_val = evaluate(model, loss_fn, val_loader, device)
        utils.log_epoch(epoch, n_epochs, train_loss, f1_val)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('F1/val', f1_val, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        
        if best_f1 < f1_val:
            best_f1 = f1_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_f1.pth'))
            logging.info("[F1] Saving model!")

    logging.info("Finished...")
writer.close()
