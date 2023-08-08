import os
import time
import argparse
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.nn as nn
from models import MMC_net
from Setup3D import Config_3D, MMC_ClassificationDataset
from utils.util import *

Precautions_msg = ' '


'''
- evaluate.py

Code that evaluates the trained model
We use the validation set we looked at during training, not the test set.

#### Manual ####
If you are using Terminal,set the path and run the code below directlypycharm


In the case of pycharm:  
Verify that [Run -> Edit Configuration -> evaluate.py] is selected
-> Go to parameters and enter below -> Run/debug after click apply
ex)Printer task 
--kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close


#### 3D Project Terminal version #### 

<Closed up Setting>
- Printer task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close --weight-num 0
- Filament task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type F --img-type close --weight-num 0
- Layer thickness task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type Q-T --img-type close --weight-num 0
- Number of shells task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type Q-S --img-type close --weight-num 0
- Device task 
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type D --img-type close --weight-num 0
- Reprint task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type R --img-type close --weight-num 0

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python evaluate.py --enet-type CFTNet --task-type D --img-type close --side-task-type P --batch-size 32 --n-epochs 20 --weight-num 0
- Multi-Task(Device & Layer thickness)
python evaluate.py --enet-type CFTNet --task-type D --img-type close --side-task-type Q-T --batch-size 32 --n-epochs 20 --weight-num 0

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python evaluate.py --enet-type CFTNet --task-type D --img-type both --batch-size 32 --n-epochs 20 --weight-num 0
- Multi-Modal-Task(Device & Printer)
python evaluate.py --enet-type CFTNet --task-type D --img-type both --side-task-type P --batch-size 32 --epoch 20 --weight-num 0
- Multi-Modal-Task(Device & Printer & Inspection data)
python evaluate.py --enet-type CFTNet --task-type D --img-type both --side-task-type P --batch-size 32 --epoch 20 --use-meta --weight-num 0

<Semi-Controlled Setting>
- Multi-Modal-Task(Printer & Number of shells)
python evaluate.py --enet-type CFTNet --task-type P --img-type both --side-task-type Q-S --batch-size 32 --epoch 20 --semi --weight-num 0
- Multi-Modal-Task(Device & Layer thickness)
python evaluate.py --enet-type CFTNet --task-type D --img-type both --side-task-type Q-T --batch-size 32 --epoch 20 --semi --weight-num 0

<Post-Processing Setting>
- Sanding-Processing-Task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close --sanding-processing --weight-num 0
- Coating-Processing-Task
python evaluate.py --enet-type CFTNet --n-epochs 20 --batch-size 32 --task-type P --img-type close --coating-processing --weight-num 0

<Fullshot Setting (Baseline only)>
- Printer task
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type P --img-type full --baseline --weight-num 0
- Filament task
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type F --img-type full --baseline --weight-num 0
- Layer thickness task 
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type Q-T --img-type full --baseline --weight-num 0
- Number of shells task 
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type Q-S --img-type full --baseline --weight-num 0
- Device task
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type D --img-type full --baseline --weight-num 0
- Reprint task
python evaluate.py --enet-type tf_efficientnet_b3_ns --n-epochs 20 --batch-size 32 --task-type R --img-type full --baseline --weight-num 0

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--enet-type', type=str, required=True)

    parser.add_argument('--batch-size', type=int, default=16, required=True)

    parser.add_argument('--task-type', type=str, default='P', required=True)
    # Task number to verify

    parser.add_argument('--weight-num', type=int, default=1, required=True)
    # Setting the weight number to evaluate with (check weight number)

    parser.add_argument('--img-type', type=str, required=True)
    # img_type = 'close', 'full', 'both'

    parser.add_argument('--DEBUG', action='store_true')

    parser.add_argument('--side-task-type', type=str, default='')

    parser.add_argument('--n-epochs', type=int, default=0)

    parser.add_argument('--use-meta', action='store_true')
    
    parser.add_argument('--baseline', action='store_true')

    parser.add_argument('--semi', action='store_true')

    parser.add_argument('--sanding-processing', action='store_true')

    parser.add_argument('--coating-processing', action='store_true')
    
    args, _ = parser.parse_known_args()
    return args



def val_epoch(model, loader, n_test=1):

    def make_log (data):
        if Setup_3d.modal:
            logits = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
            probs = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
        else:
            logits = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
            probs = torch.zeros((data[0].shape[0], Setup_3d.out_dim)).to(device)
        return logits, probs

    model.eval()

    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    OBJ_IDX = []
    MODEL_IDX = []
    PRINTER_IDX = []


    with torch.no_grad():
        for (data, target, obj_id, model_id, printer_id) in tqdm(loader):
            if Setup_3d.use_meta:
                data, meta = data
                meta = meta.to(device)
                data = list(map(lambda x: x.to(device), data))
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

            logits /= n_test
            probs /= n_test

            loss = Setup_3d.criterion(logits, target)

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            OBJ_IDX.append(obj_id.detach().cpu())
            MODEL_IDX.append(model_id.detach().cpu())
            PRINTER_IDX.append(printer_id.detach().cpu())
            val_loss.append(loss.detach().cpu().numpy())

    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    MODEL_IDX = torch.cat(MODEL_IDX).numpy()
    PRINTER_IDX = torch.cat(PRINTER_IDX).numpy()
    OBJ_IDX = torch.cat(OBJ_IDX).numpy()

    unique_idx = np.unique(OBJ_IDX).astype(np.int64)

    ConP, ConT, ConEXTRA = Setup_3d.unique_idx(TARGETS, PROBS, unique_idx, OBJ_IDX, MODEL_IDX, PRINTER_IDX)


    return ConP, ConT, ConEXTRA


def main():
    '''
        Setup_3d.task_type
        P: printer
        F: filament
        Q: quality
        Q-T: layer thickness
        Q-S: shell number
        D: device-level
        R: reprint detection

    :return:
    '''


    '''
    ####################################################
    # 3d printer dataset : dataset.get_df_3d print
    ####################################################
    '''

    Setup_3d.get_df_3dprint()

    # Recall model transforms
    Setup_3d.get_transforms()

    PROBS = []
    TARGETS = []
    EXTRA = []
    dfs = []

    # create confusion matrix
    confusion_matrix = torch.zeros(Setup_3d.out_dim, Setup_3d.out_dim)

    folds = range(Setup_3d.k_fold)

    for fold in folds:
        print(f'Evaluate data fold{str(fold)}')

        df = Setup_3d.df_train_close if Setup_3d.img_type == 'close' or Setup_3d.img_type == 'both' else Setup_3d.df_train_full

        df_valid = df[df['fold'] == fold]

        # In batch_normalization, an error can occur if batch size 1, so discard one data
        if len(df_valid) % Setup_3d.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)

        if args.DEBUG:
            df_valid = df_valid[df_valid['fold'] == fold].sample(Setup_3d.batch_size * 5)

        # Read Dataset
        dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', Setup_3d, Setup_3d.transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=Setup_3d.batch_size,
                                                   num_workers=Setup_3d.num_workers)

        side_name = "_" + Setup_3d.side_task_type if Setup_3d.side_task_type != "" else ""

        model = ModelClass(Setup_3d)
        model = model.to(device)

        model_file = os.path.join(Setup_3d.model_dir, f'{Setup_3d.weight_num}_{Setup_3d.kernel_type}_{Setup_3d.task_type}{side_name}_{Setup_3d.img_type}_{Setup_3d.n_epochs}_{Setup_3d.batch_size}{Setup_3d.type}_{fold}.pth')


        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=False)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=False)
        
        if DP:
            model = torch.nn.DataParallel(model)

        model.eval()

        '''
        ####################################################
        # evaluation function for data : val_epoch
        ####################################################
        '''
        this_PROBS, this_TARGETS, this_EXTRA = val_epoch(model, valid_loader, n_test=1)
        PROBS.append(this_PROBS)
        TARGETS.append(this_TARGETS)
        EXTRA.append(this_EXTRA)
        dfs.append(df_valid)

        for t, p in zip(this_TARGETS, this_PROBS):
                confusion_matrix[t, p] += 1

    PROBS = np.concatenate(PROBS)
    TARGETS = np.concatenate(np.concatenate(TARGETS))
    EXTRA = np.concatenate(np.concatenate(EXTRA)) #model
    ACC = PROBS == TARGETS

    Accuracy = (ACC).mean() * 100.

    if args.task_type == 'P':
        object_confusion_matrix = torch.zeros(1, 4)
        ACC_EXTRA = np.array((ACC,EXTRA))
        for i in np.unique(EXTRA):
            object_confusion_matrix[0, i] = ACC[np.where(ACC_EXTRA[1,:] == i)[0]].mean() * 100.

    content = time.ctime() + ' ' + f'Eval {Setup_3d.eval}:\nAccuracy : {Accuracy:.5f}\n'

    micro_averaged_precision = metrics.precision_score(TARGETS, PROBS, average='micro')

    print(f"\nTask name : ", Setup_3d.Task_name())
    print(f"Micro-Averaged Precision score : {micro_averaged_precision}")
    Setup_3d.acc = Accuracy
    # append the result to the end of the log file
    print(content)
    with open(os.path.join(Setup_3d.log_dir, f'log_{Setup_3d.kernel_type}.txt'), 'a') as appender:
        appender.write(content + '\n')

    Setup_3d.visualization(confusion_matrix)

    if Setup_3d.task_type == 'P':
        Setup_3d.visualization(object_confusion_matrix,obj = True)

if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # make argument
    args = parse_args()
    Setup_3d = Config_3D(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = Setup_3d.CUDA_VISIBLE_DEVICES

    ModelClass = MMC_net

    # Whether to use a multi-GPU
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    Setup_3d.criterion = nn.CrossEntropyLoss()

    # perform the main function
    main()
