import geffnet
import torch, torch.nn as nn
import timm
import torch.fft as fft

'''
class Network_name(nn.Module):
    # See the network creation section below.
    def __init__(self, net_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        # initialization
        enet_type : Network name from argument
        out_dim   : output layer size
        n_meta_dim      : mlp size (2 basic layers)
        pretrained      : Will you use a pre-trained model?

    def extract(self, x):
        # Extract the results of the base network (image deep features)
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        # Get final network results (Include fc_layer)
'''

class MMC_net(nn.Module):
    def __init__(self, Setup_3d, n_meta_dim=[512,128]):
        super(MMC_net, self).__init__()
        # efficient net Model
        self.enet = Setup_3d.select_enet()
        self.Setup_3d = Setup_3d
        self.dropouts = nn.ModuleList([
            nn.Dropout(0) for _ in range(5)
        ])

        if self.Setup_3d.use_meta:
            # adding two FC layers in inspection data experiment
            self.meta = nn.Sequential(
                nn.Linear(2, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                # swish activation function
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch_meta = n_meta_dim[1]
        else:
            in_ch_meta = 0

        if Setup_3d.kernel_type == 'CFTNet':
            self.transformer = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
            in_ch_ef = self.transformer.head.in_features + self.enet.classifier.in_features
            in_ch_cn = self.enet.classifier.in_features
            self.enet.classifier = nn.Identity()
            self.transformer.head.fc = nn.Identity()
            self.transformer.head.flatten = nn.Identity()

            self.fc_outdim1 = nn.Linear(in_ch_ef + in_ch_meta, Setup_3d.out_dim)
            self.fc_outdim1_modal = nn.Linear((in_ch_ef + in_ch_meta + in_ch_cn), Setup_3d.out_dim)
            self.fc_outdim2_multi = nn.Linear(in_ch_ef + in_ch_meta, Setup_3d.out_dim2)
            self.fc_outdim2_modal = nn.Linear((in_ch_ef + in_ch_meta + in_ch_cn), Setup_3d.out_dim2)
        else:
            try:
                self.enet.classifier.in_features
                in_ch_ef = self.enet.classifier.in_features
                self.enet.classifier = nn.Identity()
            except:
                in_ch_ef = 2048
                self.enet.head.fc = nn.Identity()
                self.enet.head.flatten = nn.Identity()


            self.fc_outdim1 = nn.Linear(in_ch_ef + in_ch_meta, Setup_3d.out_dim)
            self.fc_outdim1_modal = nn.Linear(in_ch_ef * 2 + in_ch_meta, Setup_3d.out_dim)
            self.fc_outdim2_multi = nn.Linear(in_ch_ef + in_ch_meta, Setup_3d.out_dim2)
            self.fc_outdim2_modal = nn.Linear(in_ch_ef * 2 + in_ch_meta, Setup_3d.out_dim2)



    def extract(self, x):

        if self.Setup_3d.modal:
            if self.Setup_3d.baseline:
                x1 = self.enet(x[0]).squeeze(-1).squeeze(-1)    #close
                x2 = self.enet(x[1]).squeeze(-1).squeeze(-1)    #full
                x = torch.cat((x1, x2), dim=1)                  #modal
            else:
                img_f = self.enet(x[0])
                img_fft = torch.abs(fft.fft2(x[1], dim=(2, 3), norm='ortho'))
                fft_f = self.transformer(img_fft)
                x1 = torch.cat((fft_f.flatten(1), img_f), dim=1)
                x2 = self.enet(x[2]).squeeze(-1).squeeze(-1)  # full
                x = torch.cat((x1, x2), dim=1)

            return (x, x1)
        else:
            if self.Setup_3d.baseline:
                x = self.enet(x[0]).squeeze(-1).squeeze(-1)  # close
            else:
                img_fft = torch.abs(fft.fft2(x[1], dim=(2,3),norm='ortho'))
                fft_f = self.transformer(img_fft)
                img_f = self.enet(x[0])
                x = torch.cat((fft_f, img_f), dim=1)

            return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)

        if self.Setup_3d.use_meta:
            # Using meta-data
            x_meta = self.meta(x_meta)
            if self.Setup_3d.modal:
                modal_meta = torch.cat((x[0], x_meta), dim=1)
                close_meta = torch.cat((x[1], x_meta), dim=1)
                x = (modal_meta,close_meta)
            else:
                x = torch.cat((x, x_meta), dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                if self.Setup_3d.modal:
                    out = self.fc_outdim1_modal(dropout(x[0]))
                    if self.Setup_3d.multi:
                        out2 = self.fc_outdim2_modal(dropout(x[0]))
                        out3 = self.fc_outdim2_multi(dropout(x[1]))

                    else:
                        out2 = self.fc_outdim1(dropout(x[1]))
                        out3 = 0

                else:
                    out = self.fc_outdim1(dropout(x))
                    if self.Setup_3d.multi:
                        out2 = self.fc_outdim2_multi(dropout(x))
                        out3 = 0
                    else:
                        out2 = 0
                        out3 = 0
            else:
                if self.Setup_3d.modal:
                    out += self.fc_outdim1_modal(dropout(x[0]))
                    if self.Setup_3d.multi:
                        out2 += self.fc_outdim2_modal(dropout(x[0]))
                        out3 += self.fc_outdim2_multi(dropout(x[1]))

                    else:
                        out2 += self.fc_outdim1(dropout(x[1]))

                else:
                    out += self.fc_outdim1(dropout(x))
                    if self.Setup_3d.multi:
                        out2 += self.fc_outdim2_multi(dropout(x))


        out /= len(self.dropouts)
        out2 /= len(self.dropouts)
        out3 /= len(self.dropouts)

        return out, out2, out3

sigmoid = nn.Sigmoid()

# swish activation function
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

