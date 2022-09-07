import os
import gc
import cv2
import copy
import time
import random
from PIL import Image
import wandb

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import accuracy_score

import timm

from data import prepare_loaders
from models import LandmarkRetrievalModel

# For colored terminal text
from colorama import Fore, Back, Style
g_ = Fore.GREEN
c_ = Fore.CYAN
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, CONFIG):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0 

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, (images, labels) in bar: 
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CONFIG["n_accumulate"]

        scaler.scale(loss).backward() 

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            scaler.step(optimizer)
            scaler.update()

            for p in model.parameters():
                p.grad = None 

            
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"])
    
    gc.collect() 
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0 
    running_loss = 0.0 

    TARGETS = [] 
    PREDS = [] 

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.appends(preds.view(-1).cpu().detach().numpy())
        TARGETS.append(labels.view(-1).cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

        TARGETS = np.concatenate(TARGETS)
        PREDS = np.concatenate(PREDS)
        val_acc = accuracy_score(TARGETS, PREDS)
        gc.collect() 

        return epoch_loss, val_acc


def run_training(model, optimizier, scheduelr,
                train_dataloader,
                val_dataloader,
                device,
                num_epochs, 
                run, CONFIG):

    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available:
        print(f"[INFO] using GPU: {torch.cuda.get_device_name()}\n")

        start = time.time() 
        best_model_wts = copy.deepcopy(model.state_dict())

        best_epoch_acc = 0 
        history = defaultdict(list)

        for epoch in range(1, num_epochs+1):
            gc.collect() 
            train_epoch_loss = train_one_epoch(model, optimizier, scheduelr,
                                                dataloader=train_dataloader,
                                                device=CONFIG["device"],
                                                epoch=epoch, CONFIG=CONFIG)

            val_epoch_loss, val_epoch_acc = valid_one_epoch(model,
                                            dataloader=val_dataloader, 
                                            device=CONFIG["device"], 
                                            epoch=epoch)

            history["Train_Loss"].append(train_epoch_loss)
            history["Valid_Loss"].append(val_epoch_loss)
            history["Valid_acc"].append(val_epoch_acc)

            wandb.log({"Train Loss": train_epoch_loss})
            wandb.log({"Valid Loss": val_epoch_loss})
            wandb.log({"Valid Acc": val_epoch_acc})

            print(f"Valid Acc: {val_epoch_acc}")

            if val_epoch_acc >= best_epoch_acc: 
                print(f"{c_}Validation Acc Improved ({best_epoch_acc} ---> {val_epoch_acc})")
                best_epoch_acc = val_epoch_acc
                run.summary["Best Accuracy"] = best_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = "ACC{:.4f}_epoch{:.0f}.bin".format(best_epoch_acc, epoch)
                torch.save(model.state_dict(), PATH)
                wandb.save(PATH)
                print(f"Model Save{sr_}")
            
            print()
    
    end = time.time()
    time_elapsed = end - start 

    print("Trainign Complete in {:.0f}h {:.f}m {:.0f}s".format(time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60))

    print("Best ACC: {:.4f}".format(best_epoch_acc))

    model.load_state_dict(best_model_wts)

    return model, history



def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], 
                                                             T_mult=1, eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def set_seed(seed=42): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 

    os.environ["PYTHONHASHSEED"] = str(seed)
    

if __name__ == "__main__":

    CONFIG = dict(
    seed = 42, 
    model_name = "tf_mobilenetv3_small_100", 
    train_batch_size = 1, 
    valid_batch_size = 1, 
    img_size = 224, 
    epochs = 3, 
    learning_rate = 5e-4, 
    scheduler = None, 
    weight_decay = 1e-6, 
    n_accumulate = 1, 
    n_folds = 5, 
    num_classes = 81313, 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
    competition = "GOOGL", 
    _wandb_kernel = "deb"
)

set_seed(CONFIG['seed'])

train_dataloder, valid_dataloader = prepare_loaders(CONFIG)

model = LandmarkRetrievalModel(CONFIG["model_name"], CONFIG["num_classes"])
model.to(CONFIG['device'])

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)

run = wandb.init(project='GLRet2021', 
                 config=CONFIG,
                 job_type='Train',
                 anonymous='must')

                 
model, history = run_training(model, optimizer, 
                            scheduler,
                            train_dataloder,
                            valid_dataloader, 
                            CONFIG["device"],
                            CONFIG["epochs"], 
                            run, 
                            CONFIG)