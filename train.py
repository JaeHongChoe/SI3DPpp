import os
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from utils.util import *
from models import MMC_net
from Setup3D import Config_3D, MMC_ClassificationDataset

Precautions_msg = '(Precautions for use) ---- \n'

'''
- train.py

Code containing the entire process

#### Manual ####
If you are using Terminal,set the path and run the code below directlypycharm

In the case of pycharm:  
Verify that [Run -> Edit Configuration -> train.py] is selected
-> Go to parameters and enter below -> Run/debug after click apply
ex)Printer task 
--kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

*** def parse_args(): There is all the information about the execution parameters.  
*** def run(): A function that contains the whole process of learning. You can improve performance by applying various tricks here.
** def main(): Run after distributing the data divided by the fold to the [def run].
* def train_epoch(), def val_epoch() : Correction after understanding completely

 MMCLab, 허종욱, 2020python 



### 3D Project Terminal ###

<Closed up Setting>
- Printer task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close
- Filament task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type F --img-type close
- Layer thickness task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type Q-T --img-type close
- Number of shells task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type Q-S --img-type close
- Device task 
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type D --img-type close
- Reprint task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type R --img-type close

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python train.py --enet-type CFTNet --task-type D --img-type close --side-task-type P --batch-size 32 --n-epochs 20
- Multi-Task(Device & Layer thickness)
python train.py --enet-type CFTNet --task-type D --img-type close --side-task-type Q-T --batch-size 32 --n-epochs 20

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python train.py --enet-type CFTNet --task-type D --img-type both --batch-size 32 --n-epochs 20
- Multi-Modal-Task(Device & Printer)
python train.py --enet-type CFTNet --task-type D --img-type both --side-task-type P --batch-size 32 --epoch 20
- Multi-Modal-Task(Device & Printer & Inspection data)
python train.py --enet-type CFTNet --task-type D --img-type both --side-task-type P --batch-size 32 --epoch 20 --use-meta

<Semi-Controlled Setting>
- Multi-Modal-Task(Printer & Number of shells)
python train.py --enet-type CFTNet --task-type P --img-type both --side-task-type Q-S --batch-size 32 --epoch 20 --semi
- Multi-Modal-Task(Device & Layer thickness)
python train.py --enet-type CFTNet --task-type D --img-type both --side-task-type Q-T --batch-size 32 --epoch 20 --semi

<Post-Processing Setting>
- Sanding-Processing-Task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close --sanding-processing
- Coating-Processing-Task
python train.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close --coating-processing

<Fullshot Setting (Baseline only)>
- Printer task
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type P --img-type full --baseline
- Filament task
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type F --img-type full --baseline
- Layer thickness task 
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type Q-T --img-type full --baseline
- Number of shells task 
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type Q-S --img-type full --baseline
- Device task
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type D --img-type full --baseline
- Reprint task
python train.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type R --img-type full --baseline

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--enet-type', type=str, required=True)
    # Network name to apply to learning
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--task-type', type=str, default='P', required=True)
    # Setting the task type to experiment with

    parser.add_argument('--side-task-type', type=str, default='')
    # Setting the multi-task type to experiment with

    parser.add_argument('--DEBUG', action='store_true')
    # Parameters for Debugging (Hold the experimental epoch at 5)

    parser.add_argument('--use-meta', action='store_true')
    # Whether to use inspection data in addition to the original

    parser.add_argument('--baseline', action='store_true')
    # Setting the network structure (our previous network)

    parser.add_argument('--semi', action='store_true')
    # Setting Semi-controlled task to experiment with

    parser.add_argument('--sanding-processing', action='store_true')
    # Setting sanding-processing task to experiment with

    parser.add_argument('--coating-processing', action='store_true')
    # Setting coating-processing task to experiment with

    parser.add_argument('--img-type', type=str, required=True)
    # Whether to use {close, full, both} image data

    parser.add_argument('--batch-size', type=int, default=16, required=True)
    # batch size

    parser.add_argument('--n-epochs', type=int, default=30, required=True)
    # number of epochs

    parser.add_argument('--weight-num', type=int, default=0)

    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer, Setup_3d):
    model.train()
    train_loss = []
    bar = tqdm(loader)

    for i, (data, target) in enumerate(bar):
        optimizer.zero_grad()

        if Setup_3d.use_meta:
            data, meta = data
            meta = meta.to(device)
            data = list(map(lambda x: x.to(device), data))
            target = list(map(lambda x: x.to(device), target)) if Setup_3d.multi else target.to(device)
            logits1, logits2, logits3 = model(data, meta)
        else:
            data = list(map(lambda x: x.to(device), data))
            target = list(map(lambda x: x.to(device), target)) if Setup_3d.multi else target.to(device)
            logits1, logits2, logits3 = model(data)

        loss = Setup_3d.select_loss(logits1, logits2, logits3, target)
        loss.backward()

        # gradient accumulation (When memory is low)
        if Setup_3d.accumulation_step:
            if (i + 1) % Setup_3d.accumulation_step == 0:
                optimizer.step()
        else:
            optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def val_epoch(model, loader, Setup_3d, n_test=1):
    '''

    Output:
    val_loss, acc, TARGETS, PROBS
    '''

    def make_log(data):
        if Setup_3d.modal:
            logits = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
            probs = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
        else:
            logits = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
            probs = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
        return logits, probs

    model.eval()

    val_loss = []
    PROBS = []
    TARGETS = []
    OBJ_IDX = []

    with torch.no_grad():
        for i, (data, target, obj_id, model_id, printer_id) in enumerate(tqdm(loader)):
            if Setup_3d.use_meta:
                data, meta = data
                meta = meta.to(device)
                data = list(map(lambda x: x.to(device), data)) if Setup_3d.modal else data.to(device)
                target = target[0].to(device) if Setup_3d.multi else target.to(device)
                logits, probs = make_log(data)
                for I in range(n_test):
                    l, l_2, l_3 = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data = list(map(lambda x: x.to(device), data))
                target = target[0].to(device) if Setup_3d.multi else target.to(device)
                logits, probs = make_log(data)
                for I in range(n_test):
                    l, l_2, l_3 = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)

            loss = Setup_3d.criterion(logits, target)

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            OBJ_IDX.append(obj_id)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    OBJ_IDX = torch.cat(OBJ_IDX).numpy()
    unique_idx = np.unique(OBJ_IDX).astype(np.int64)

    for u_id in unique_idx.tolist():
        res_list = np.where(OBJ_IDX == u_id)
        mean_prob = PROBS[res_list].mean(axis=0)
        PROBS[res_list] = mean_prob

    # accuracy
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    return val_loss, acc, TARGETS, PROBS.argmax(1)


def run(fold, Setup_3d):
    '''
    Learning progress main function

    :param fold: The partition number to be used for value in cross-validation
    :param df: Full Data List for DataFrame Learning
    :param meta_features, n_meta_features: Whether to use additional information other than images
    :param transforms_train, transforms_val: Dataset transform function
    '''

    df = Setup_3d.df_train_close if Setup_3d.img_type == 'close' or Setup_3d.img_type == 'both' else Setup_3d.df_train_full

    if args.DEBUG:
        Setup_3d.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(Setup_3d.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(Setup_3d.batch_size * 5)

    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        #
        if len(df_train) % Setup_3d.batch_size == 1:
            df_train = df_train.sample(len(df_train) - 1)
        if len(df_valid) % Setup_3d.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid) - 1)

    # Read Dataset
    dataset_train = MMC_ClassificationDataset(df_train, 'train', Setup_3d
                                            ,Setup_3d.transforms_train, Setup_3d.transforms_fft)

    dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', Setup_3d
                                              , Setup_3d.transforms_val)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=Setup_3d.batch_size,
                                               sampler=RandomSampler(dataset_train), num_workers=Setup_3d.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=Setup_3d.batch_size,
                                               num_workers=3, pin_memory=True)


    side_name = "_" + Setup_3d.side_task_type if Setup_3d.side_task_type != "" else ""

    acc_max = 0.

    model = ModelClass(Setup_3d)

    model_file = os.path.join(Setup_3d.model_dir,
                              f'{Setup_3d.weight_num}_{Setup_3d.kernel_type}_{Setup_3d.task_type}{side_name}_{Setup_3d.img_type}_{Setup_3d.n_epochs}_{Setup_3d.batch_size}{Setup_3d.type}_{fold}.pth')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Setup_3d.init_lr)

    if DP:
        model = nn.DataParallel(model)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, Setup_3d.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)

    for epoch in range(1, Setup_3d.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer, Setup_3d)

        val_loss, acc, targets, probs = val_epoch(model, valid_loader, Setup_3d)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}'

        print(content)
        with open(os.path.join(Setup_3d.log_dir, f'log_{Setup_3d.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()  # bug workaround

        if acc > acc_max:
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), model_file)
            acc_max = acc


def main():
    '''
    ####################################################
    # 3d printer dataset : dataset.get_df_3d print
    ####################################################
    '''

    Setup_3d.get_df_3dprint()

    # Recall model transforms
    Setup_3d.get_transforms()

    folds = range(Setup_3d.k_fold)

    for fold in folds:
        run(fold, Setup_3d)


if __name__ == '__main__':
    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # make argument
    args = parse_args()
    Setup_3d = Config_3D(args)

    os.makedirs(Setup_3d.model_dir, exist_ok=True)
    os.makedirs(Setup_3d.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = Setup_3d.CUDA_VISIBLE_DEVICES

    Setup_3d.weight_num = int(len(os.listdir(Setup_3d.model_dir)) / 4)

    ModelClass = MMC_net

    # Whether to use a multi-GPU
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # Random seed settings for experimental reproduction
    set_seed(2359)
    device = torch.device('cuda')
    Setup_3d.criterion = nn.CrossEntropyLoss()

    # perform the main function
    main()