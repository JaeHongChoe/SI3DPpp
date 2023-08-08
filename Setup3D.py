import pandas as pd
import os
import albumentations
import numpy as np
from torch.utils.data import Dataset
import random
import geffnet
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import cv2
import timm

class Config_3D():
    def __init__(self, args):
        def set_outdim(type):
            if type == 'P':
                return 9
            elif type == 'F':
                return 8
            elif type == 'Q':
                return 3
            elif type == 'Q-T':
                return 5
            elif type == 'Q-S':
                return 3
            elif type == 'D':
                return 25
            elif type == 'R':
                return 2
            else:
                return 0

        self.k_fold = 4
        self.data_dir = './data/'
        self.data_folder = 'SI3DPpp/'
        self.CUDA_VISIBLE_DEVICES = '0'
        self.image_size = 224
        self.log_dir = './logs'
        self.model_dir = './weights'
        self.accumulation_step = 1
        self.num_workers = 4
        self.init_lr = 1e-5
        self.kernel_type = ''
        self.eval = 'best'
        self.use_meta = args.use_meta
        self.semi_control = args.semi
        self.SP = args.sanding_processing
        self.CP = args.coating_processing

        self.enet_type = args.enet_type
        self.task_type = args.task_type
        self.side_task_type = args.side_task_type
        self.img_type = args.img_type
        self.out_dim = set_outdim(self.task_type)
        self.out_dim2 = set_outdim(self.side_task_type)
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num
        self.baseline = args.baseline
        self.type = ''
        #self.gpu_num = args.gpu_num

        self.df_train_close = None
        self.df_train_full = None
        self.df_test = None
        self.transforms_train = None
        self.transforms_val = None
        self.transforms_fft = None
        self.criterion = None
        self.modal = True if args.img_type == 'both' else False
        self.multi = True if self.out_dim2 else False
        self.acc = None


    def select_enet(self):
        if self.baseline:
            if self.enet_type == 'tf_mixnet_l':
                self.kernel_type = 'mixnet_l'
                return geffnet.create_model(self.enet_type, pretrained=True)
            elif self.enet_type == 'resnetv2':
                self.kernel_type = 'resnetv2'
                return timm.create_model('resnetv2_101',pretrained=True)
            elif self.enet_type == 'regnetz':
                self.kernel_type = 'regnetz'
                return timm.create_model('regnetz_e8', pretrained=True)
            elif 'efficientnet' in self.enet_type:
                self.kernel_type = 'EF'+self.enet_type.split('_')[2]
                return geffnet.create_model(self.enet_type, pretrained=True)
            else:
                raise NotImplementedError()
        else:
            if 'CFTNet' == self.enet_type:
                self.kernel_type = 'CFTNet'
                return geffnet.create_model('tf_efficientnet_b3', pretrained=True)
            else:
                raise NotImplementedError()

    def select_loss(self, logits1, logits2, logits3, target):
        alpha = 0.9
        beta = 0.1

        if self.modal:
            if self.multi:
                target1 , target2 = target[0], target[1]
                loss1_cf_d = self.criterion(logits1, target1)       #modal / main
                loss2_cf_q = self.criterion(logits2, target2)       #modal / side
                loss3_cf_q = self.criterion(logits3, target2)       #close / side

                loss = (loss1_cf_d + loss3_cf_q) * alpha + beta * (loss2_cf_q)
            else:
                loss = self.criterion(logits1, target) * alpha + beta * self.criterion(logits2, target)
        else:
            if self.multi:
                target1, target2 = target[0], target[1]
                loss = self.criterion(logits1, target1) * alpha + beta * self.criterion(logits2, target2)
            else:
                loss = self.criterion(logits1, target)

        return loss

    def get_df_3dprint(self):
        def make_df(data_dir,data_folder,task_type,img_type):
            if task_type == 'D': task_type = '_device'
            elif task_type == 'R': task_type = '_reprint'
            else: task_type = ''


            if img_type == 'full':
                df = pd.read_csv(os.path.join(data_dir, data_folder, f'train_{img_type}.csv'))
            else:
                df = pd.read_csv(os.path.join(data_dir, data_folder, f'train_{img_type}{task_type}.csv'))

            df['filepath'] = df['image_name'].apply(
                lambda x: os.path.join(data_dir, f'{data_folder}train_{img_type}', x))

            # Original data=0, Meta data=1
            df['is_ext'] = 0
            return df

        df_train_close = make_df(self.data_dir,self.data_folder,self.task_type,'close')
        df_train_full = make_df(self.data_dir, self.data_folder, self.task_type,'full')

        # test data
        df_test = pd.read_csv(os.path.join(self.data_dir, self.data_folder, 'test.csv'))
        df_test['filepath'] = df_test['image_name'].apply(
            lambda x: os.path.join(self.data_dir, f'{self.data_folder}test', x))  # f'{x}.jpg'


        def target_set(df,task_type,side_task = False):
            if task_type == 'P': df_detail = df.printer; task_type = 'printer'
            elif task_type == 'F': df_detail = df.filament; task_type = 'filament'
            elif task_type == 'Q': df_detail = df.quality; task_type = 'quality'
            elif task_type == 'Q-T': df_detail = df.thickness; task_type = 'thickness'
            elif task_type == 'Q-S': df_detail = df.shell_num; task_type = 'shell_num'
            elif task_type == 'D': df_detail = df.num; task_type = 'num'
            elif task_type == 'R': df_detail = df.reprint; task_type = 'reprint'
            else: return df

            if side_task:
                target2idx2 = {d: idx for idx, d in enumerate(sorted(df_detail.unique()))}
                df['target2'] = df[task_type].map(target2idx2)
            else:
                self.target2idx = {d: idx for idx, d in enumerate(sorted(df_detail.unique()))}
                df['target'] = df[task_type].map(self.target2idx)
                self.printer2idx = {d: idx for idx, d in enumerate(sorted(df.printer.unique()))}
                self.model2idx = {d: idx for idx, d in enumerate(sorted(df.model.unique()))}
                df['printer_id'] = df['printer'].map(self.printer2idx)
                df['model_id'] = df['model'].map(self.model2idx)
            return df

        df_train_close = target_set(df_train_close,self.task_type)
        df_train_full = target_set(df_train_full, self.task_type)


        df_train_close = target_set(df_train_close, self.side_task_type,True)
        df_train_full = target_set(df_train_full, self.side_task_type,True)

        self.df_train_close = df_train_close
        self.df_train_full = df_train_full
        self.df_test = df_test


    def get_transforms(self):
        image_size = self.image_size

        transforms_train = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            albumentations.MedianBlur(blur_limit=3, p=0.75),
            albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            albumentations.CLAHE(clip_limit=4.0, p=0.4),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(image_size, image_size),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1,
                                  p=0.7),
            albumentations.Normalize(),
        ])

        transforms_fft = albumentations.Compose([
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
        ])

        transforms_val = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize()
        ])

        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.transforms_fft = transforms_fft

    def unique_idx(self, target, probs, unique_idx, obj_idx, model_idx, printer_idx):
        ConT = []
        ConP = []
        ConEXTRA = []

        for u_id in unique_idx.tolist():
            res_list = np.where(obj_idx == u_id)
            ConT.append(np.unique(target[res_list]))
            if self.task_type == 'P':
                ConEXTRA.append(np.unique(model_idx[res_list]))
            else: ConEXTRA.append(np.unique(printer_idx[res_list]))
            mean_prob = probs[res_list].mean(axis=0)
            ConP.append(mean_prob.argmax())
            probs[res_list] = mean_prob

        return ConP,ConT, ConEXTRA

    def visualization(self, confusion_matrix, obj=False):
        label = list(self.target2idx)
        obj_label = list(self.model2idx)

        if self.task_type == 'P':
            title = 'Object Model' if obj else 'Printer Model'
        elif self.task_type == 'F':
            title = 'Filament Model'
        elif self.task_type == 'Q':
            title = 'Quality Model'
        elif self.task_type == 'Q-T':
            title = 'Layer Thickness'
        elif self.task_type == 'Q-S':
            title = 'Number of Shells'
        elif self.task_type == 'D':
            title = 'Device Model'
        elif self.task_type == 'R':
            title = 'Reprint Model'

        mpl.style.use('seaborn')

        if obj:
            confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=[self.img_type], columns=obj_label)
        else:
            total = np.sum(confusion_matrix.numpy(), axis=1)
            confusion_matrix = confusion_matrix / total[:, None]
            confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=label, columns=label)

        fig = plt.figure(figsize=(12, 9))
        plt.clf()

        ax = fig.add_subplot(111)
        ax.set_aspect(1)

        cmap = sns.cubehelix_palette(light=0.95, dark=0.08, as_cmap=True)
        ax = sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 10}, fmt='.2f', linewidths=0.3, cmap=cmap, clip_on=False)

        if obj:
            plt.xticks(np.arange(len(obj_label)) + 0.5, obj_label, size=25, rotation='vertical')
            plt.yticks(np.arange(len([self.img_type])) + 0.5, [self.img_type], size=25, rotation='horizontal')
        else:
            plt.xticks(np.arange(len(label)) + 0.5, label, size=25, rotation='vertical')
            plt.yticks(np.arange(len(label)) + 0.5, label, size=25, rotation='horizontal')

        ax.axhline(y=0, color='k', linewidth=1)
        ax.axhline(y=len(confusion_matrix), color='k', linewidth=2)
        ax.axvline(x=0, color='k', linewidth=1)
        ax.axvline(x=len(confusion_matrix), color='k', linewidth=2)


        plt.savefig(os.path.join('./graph',
                                 f'{self.weight_num}_{title}_({self.task_type}+{self.side_task_type})_{self.img_type}.png'),
                    dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    def Task_name(self):

        if self.img_type == 'close': image_name = 'Closeshot'
        elif self.img_type == 'full': image_name = 'Fullshot'
        else: image_name = 'Modal(Close & Full)'

        if self.task_type == 'P': task_name = 'Main-task : Printer'
        elif self.task_type == 'F': task_name = 'Main-task : Filament'
        elif self.task_type == 'Q-S': task_name = 'Main-task : Shell Num'
        elif self.task_type == 'Q-T': task_name = 'Main-task : Layer Thickness'
        elif self.task_type == 'D': task_name = 'Main-task : Device'
        elif self.task_type == 'R': task_name = 'Main-task : Reprint'

        if self.side_task_type == 'P': side_task_name = 'Side-task : Printer'
        elif self.side_task_type == 'F': side_task_name = 'Side-task : Filament'
        elif self.side_task_type == 'Q-S': side_task_name = 'Side-task: Shell Num'
        elif self.side_task_type == 'Q-T': side_task_name = 'Side-task : Layer Thickness'
        elif self.side_task_type == 'D': side_task_name = 'Side-task : Device'
        elif self.side_task_type == 'R': side_task_name = 'Side-task : Reprint'
        else: side_task_name = ''

        name = image_name + ' / ' + task_name + ' / ' + side_task_name

        return name


class MMC_ClassificationDataset(Dataset):
    '''
    MMC_ClassificationDataset 클래스
    Dataset class for image classification
        class Dataset_def_name(Dataset):
            def __init__(self, csv, mode, meta_features, transform=None):
                # Dataset initialization

            def __len__(self):
                # return dataset length
                return self.csv.shape[0]

            def __getitem__(self, index):
                # Returns the image corresponding to the index
    '''

    def __init__(self, csv, mode, Setup_3d, transform = None, tt=None):
        self.Setup_3d = Setup_3d
        self.mode = mode # train / valid
        self.csv = csv.reset_index(drop=True)
        self.csv2 = Setup_3d.df_train_full.reset_index(drop=True)
        self.transform = transform
        self.transform_fft = tt
        self.task_type = Setup_3d.task_type
        self.side_task = Setup_3d.side_task_type
        self.img_type = Setup_3d.img_type

        if self.Setup_3d.semi_control:
            self.Setup_3d.type = '_semi'
            if self.mode == 'train':
                self.csv = self.csv[self.csv.quality != 'MQ']
                self.csv2 = self.csv2[self.csv2.quality != 'MQ']
            else:
                self.csv = self.csv[self.csv.quality == 'MQ']
                self.csv2 = self.csv2[self.csv2.quality == 'MQ']

        if self.Setup_3d.SP:
            self.Setup_3d.type = '_SP'
            # sanding-processing
            if self.mode == 'train':
                self.csv = self.csv[self.csv.processing != 'coating']
                self.csv2 = self.csv2[self.csv2.processing != 'coating']
            else:
                self.csv = self.csv[self.csv.processing == 'sandpaper']
                self.csv2 = self.csv2[self.csv2.processing == 'sandpaper']
        elif self.Setup_3d.CP:
            self.Setup_3d.type = '_CP'
            # coating-processing
            if self.mode == 'train':
                self.csv = self.csv[self.csv.processing != 'sandpaper']
                self.csv = self.csv2[self.csv2.processing != 'sandpaper']
            else:
                self.csv = self.csv[self.csv.processing == 'coating']
                self.csv2 = self.csv2[self.csv2.processing == 'coating']
        else:
            # base
            self.csv = self.csv[self.csv.processing == 'base']
            self.csv2 = self.csv2[self.csv2.processing == 'base']

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        # Extraction target and object id
        if self.img_type == 'both':
            if self.mode == 'valid':
                positive_rand_filepath = random.choice([id for id in self.csv2.loc[self.csv2['obj_id'] == row.obj_id].filepath])
            elif self.side_task:
                positive_rand_filepath = random.choice([id for id in self.csv2.loc[self.csv2['target2'] == row.target2].filepath])
            else:
                positive_rand_filepath = random.choice([id for id in self.csv2.loc[self.csv2['obj_id'] == row.obj_id].filepath])
            image_main = self._read_image_row(row.filepath)
            image_side = self._read_image_row(positive_rand_filepath) # bring full image random path

            data = [image_main, image_side]
            data = sum(data, [])
        else:
            image_main = self._read_image_row(row.filepath)
            data = image_main

        obj_id = torch.tensor(row['obj_id'])
        model_id = torch.tensor(row['model_id'])
        printer_id = torch.tensor(row['printer_id'])

        if self.Setup_3d.multi:
            target1 = torch.tensor(self.csv.iloc[index].target).long()
            target2 = torch.tensor(self.csv.iloc[index].target2).long()
            target = (target1,target2)
        else:
            target = torch.tensor(self.csv.iloc[index].target).long()


        if self.mode == 'valid':
            if self.Setup_3d.use_meta:
                data = (data, torch.tensor(self.csv.iloc[index][[ 'shell_num', 'thickness']]).float())
                return data, target, obj_id, model_id, printer_id
            else:
                return data, target, obj_id, model_id, printer_id
        else:
            if self.Setup_3d.use_meta:
                data = (data, torch.tensor(self.csv.iloc[index][['shell_num', 'thickness']]).float())
                return data, target
            else:
                return data, target

    def _read_image_row(self, filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Apply image tranform
        if self.transform is not None:
            if self.transform_fft is None:
                self.transform_fft = self.transform
            res = self.transform(image=image)
            res_fft = self.transform_fft(image=image)

            image_fft = res_fft['image'].astype(np.float32)
            image = res['image'].astype(np.float32)

            image = image.transpose(2, 0, 1)
            image = torch.tensor(image).float()
            image_fft = image_fft.transpose(2, 0, 1)
            image_fft = torch.tensor(image_fft).float()

            image = [image, image_fft]

        else:
            image = image.astype(np.float32)

            image = image.transpose(2, 0, 1)
            image = torch.tensor(image).float()

        return image
