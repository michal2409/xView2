import os
import json
import utils
import torch
import segm as smp
import logging
import numpy as np
import segm as smp
from evaluate import evaluate
import models.data_loader as data_loader
import torchvision.transforms.functional as TF

def train_epoch(model, dataloader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 130], gamma=0.1)

    model.train()
    total, correct = 0, 0
    acc, loss_avg, f1 = utils.RunningAverage(), utils.RunningAverage(), utils.RunningF1()
    for X, (y_img, y_aug) in dataloader:
        X, y_img, y_aug = X.to(device), y_img.type(torch.LongTensor).to(device).squeeze(), y_aug.to(device)
        optimizer.zero_grad()
        pred_img, pred_aug = model(X.float())
        loss = 0.7*loss_fn(pred_img, y_img) + 0.3*loss_fn(pred_aug, y_aug)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())

        _, predicted = torch.max(pred_img.data, 1)
        correct, total = (predicted == y_img).sum().item(), y_img.shape[0]*y_img.shape[1]*y_img.shape[2]
        acc.update(100*correct/total)
        f1.update(predicted.squeeze().cpu().numpy(), y_img.squeeze().cpu().numpy())
    scheduler.step()
    return acc(), loss_avg(), f1()

if __name__ == '__main__':
    utils.set_seed()
    device, params, output_dir = utils.prepare_env()
    with open(os.path.join(output_dir, 'params.json'), 'w') as outfile:
        json.dump(params, outfile)

    logging.info("Fetching dataloaders...")
    train_loader = data_loader.fetch_dataloader(params['train_data'], params['batch_size'], 'Training', params)
    val_loader = data_loader.fetch_dataloader(params['val_data'], params['batch_size'], 'Validation')

    logging.info("Buidling model...")
    model = smp.Unet('resnet101', classes=2, encoder_weights='imagenet', activation='softmax').to(device)
    logging.info("Finished building...")

    if params['pretrained_model'] is not None:
        logging.info("Loading pretrained weights")
        model = utils.load_model(model, params['pretrained_model'])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    n_epochs = params['epochs']
    best_val_loss, best_f1 = float('Inf'), 0.0
    logging.info("Training...")
    for epoch in range(n_epochs):
        train_acc, train_loss, train_f1 = train_epoch(model, train_loader, device)
        val_acc, val_loss, f1_val = evaluate(model, val_loader, device, params)
        utils.log_epoch(epoch, n_epochs, train_loss, train_acc, train_f1, val_loss, val_acc, f1_val)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            utils.save_model(model, output_dir, 'best_loss.pth')
            logging.info("[Loss] Saving model!")

        if best_f1 < f1_val:
            best_f1 = f1_val
            utils.save_model(model, output_dir, 'best_f1.pth')
            logging.info("[F1] Saving model!")

    utils.save_model(model, output_dir, 'last.pth')
    logging.info("[Last] Saving model!")
    logging.info("Finished...")
