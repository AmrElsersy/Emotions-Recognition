"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse
import logging
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

from model.model import Mini_Xception
from model.depthwise_conv import SeparableConv2D
from Utils.dataset import create_train_dataloader, create_val_dataloader

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint/logging', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data', help='root path of augumented WFLW dataset')
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/weights.pth1.tar',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoint/model_weights', help='save checkpoint path')    
    parser.add_argument('--savefreq', type=int, default=5, help="save weights each freq num of epochs")
    parser.add_argument('--logdir', type=str, default='checkpoint/logging', help='logging')    
    parser.add_argument("--lr_patience", default=40, type=int)
    args = parser.parse_args()
    return args
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
# logging
logging.basicConfig(
format='[%(message)s',
level=logging.INFO,
handlers=[logging.FileHandler(args.logdir, mode='w'), logging.StreamHandler()])
# tensorboard
writer = tensorboard.SummaryWriter(args.tensorboard)

def main():
    # ========= dataloaders ===========
    train_dataloader = create_train_dataloader(root=args.datapath, batch_size=args.batch_size)
    test_dataloader = create_val_dataloader(root=args.datapath, batch_size=args.batch_size)
    start_epoch = 0
    # ======== models & loss ========== 
    mini_xception = Mini_Xception()
    loss = nn.CrossEntropyLoss()
    # ========= load weights ===========
    if args.resume:
        checkpoint = torch.load(args.pretrained)
        mini_xception.load_state_dict(checkpoint['mini_xception'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'\tLoaded checkpoint from {args.pretrained}\n')
        time.sleep(1)
    else:
        print("******************* Start training from scratch *******************\n")
        # time.sleep(5)
    # =========== optimizer =========== 
    optimizer = torch.optim.Adam(mini_xception.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # =========== train / validate ===========
        # train_loss = train_one_epoch(mini_xception, loss, optimizer, train_dataloader, epoch)
        val_loss, accuracy, percision, recall = validate(mini_xception, loss, test_dataloader, epoch)
        scheduler.step(val_loss)
        logging.info(f"\ttraining epoch={epoch} .. train_loss={train_loss}")
        logging.info(f"\tvalidation epoch={epoch} .. val_loss={val_loss}")
        logging.info(f'\tAccuracy = {accuracy} .. Percision = {percision} .. Recall = {recall}')
        time.sleep(2)
        # ============= tensorboard =============
        writer.add_scalar('train_loss',train_loss, epoch)
        writer.add_scalar('val_loss',val_loss, epoch)
        # ============== save model =============
        if epoch % args.savefreq == 0:
            checkpoint_state = {
                'mini_xception': mini_xception.state_dict(),
                "epoch": epoch
            }
            savepath = os.path.join(args.savepath, f'weights_epoch_{epoch}.pth.tar')
            torch.save(checkpoint_state, savepath)
            print(f'\n\t*** Saved checkpoint in {savepath} ***\n')
            time.sleep(2)
    writer.close()

def train_one_epoch(model, criterion, optimizer, dataloader, epoch):
    model.train()
    model.to(device)
    loss = 0
    print(model)
    for images, labels in tqdm(dataloader):

        images = images.to(device) # (batch, 1, 48, 48)
        labels = labels.to(device) # (batch,)

        emotions = model(images)
        # from (batch, 7, 1, 1) to (batch, 7)
        emotions = torch.squeeze(emotions)

        loss = criterion(emotions, labels)
        print(f'training @ epoch {epoch} .. loss = {round(loss.item(),3)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def validate(model, criterion, dataloader, epoch):
    model.eval()
    model.to(device)
    losses = []
    TP = 0

    total_pred = []
    total_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            mini_batch = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            emotions = model(images)
            emotions = torch.squeeze(emotions)
            emotions = emotions.reshape(mini_batch, -1)

            loss = criterion(emotions, labels)            
            losses.append(loss.cpu().item())

            # softmax = nn.Softmax()
            # logsoft = nn.LogSoftmax()
            # emotions_soft = softmax(emotions)
            # emotions_logsoft = logsoft(emotions)
            # l2 = nn.NLLLoss()
            # loss2 = l2(emotions_logsoft, labels)
            # print(f'softmax {emotions_soft}')
            # print(f'log softmax {emotions_logsoft}')
            # print(f'NLL {loss2}')
            # print(f'emotions {emotions}\n')
            # # ============== Evaluation ===============

            # index of the max value of each sample (shape = (batch,))
            _, indexes = torch.max(emotions, axis=1)
            # print(indexes.shape, labels.shape)
            total_pred.extend(indexes.cpu().detach().numpy())
            total_labels.extend(labels.cpu().detach().numpy())

            # print(f'validation loss = {round(loss.item(),3)}')

        val_loss = np.mean(losses).item()
        percision = precision_score(total_labels, total_pred, average='macro')
        recall = recall_score(total_labels, total_pred, average='macro')
        accuracy = accuracy_score(total_labels, total_pred)

        val_loss, accuracy, percision, recall = round(val_loss,3), round(accuracy,3), round(percision,3), round(recall,3)
        print(f'Val loss = {val_loss} .. Accuracy = {accuracy} .. Percision = {percision} .. Recall = {recall}')
        return val_loss, accuracy, percision, recall

if __name__ == "__main__":
    main()
    # total_labels = [0, 1, 2, 0]
    # total_pred =   [0, 2, 1, 2]
    # avg = None
    # avg = 'macro'
    # percision = precision_score(total_labels, total_pred, average=avg)
    # recall = recall_score(total_labels, total_pred, average=avg)
    # accuracy = accuracy_score(total_labels, total_pred)
    # print(percision, recall, accuracy)
