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

DAMAGE = False

def train_epoch(model, optimizer, dataloader, scheduler, device):
    t0 = timeit.default_timer()
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    model.train()
    loss_avg, f1 = utils.RunningAverage(), utils.RunningF1()
    for X, y_true in dataloader:
        X, y_true = X.to(device), y_true.to(device)
        y_pred = model(X)
        
        if DAMAGE:
            loss = utils.get_damage_loss(y_pred, y_true[:, :, :, :5], loss_fn)
        else:
            loss = loss_fn(y_pred.squeeze(), y_true.type_as(y_pred))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        loss_avg.update(loss.item())
        
        if DAMAGE:
            y_true = y_true[:, :, :, 5]
            _, predicted = torch.max(y_pred.data, 1)
        else:
            predicted = (y_pred[:, 0] > 0).long()
        f1.update(predicted.squeeze().cpu().numpy(), y_true.squeeze().cpu().numpy())
    
    if amp._amp_state.loss_scalers[0]._unskipped != 0:
        scheduler.step()
    elapsed = timeit.default_timer() - t0
    return loss_avg(), f1(), elapsed

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
        
    if params['pretrained_model'] is not None:
        logging.info("Loading pretrained weights...")
        model = utils.load_model(model, params['pretrained_model'])
        
#     optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    logging.info("Finished building...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params['epochs'], eta_min=1e-6)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    n_epochs = params['epochs']
    best_val_loss, best_f1 = float('Inf'), 0.0
    logging.info("Training...")
    for epoch in range(n_epochs):
        train_loss, train_f1, train_elaps = train_epoch(model, optimizer, train_loader, scheduler, device)
        val_loss, f1_val, eval_elaps = evaluate(model, val_loader, device, params)
        utils.log_epoch(epoch, n_epochs, train_loss, train_f1, val_loss, f1_val)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', f1_val, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Time/train', train_elaps, epoch)
        writer.add_scalar('Time/val', eval_elaps, epoch)
        
        if best_f1 < f1_val:
            best_f1 = f1_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_f1.pth'))
            logging.info("[F1] Saving model!")

    torch.save(model.state_dict(), os.path.join(output_dir, 'last.pth'))
    logging.info("[Last] Saving model!")
    logging.info("Finished...")
writer.close()
