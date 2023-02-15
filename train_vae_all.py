"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from tqdm import tqdm

from dataset import get_df, get_transforms, MelanomaDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

"""
A Convolutional Variational Autoencoder
"""
class Res_Block(nn.Module):

    def __init__(self,imgChannels,Gate_para=0.1,if_HR_LR = False):
        super(Res_Block,self).__init__()

        self.conv1 = nn.Conv2d(imgChannels, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, imgChannels, kernel_size=1, stride=1)
        self.conv_concat = nn.Conv2d(2*imgChannels,32,kernel_size=1,stride=1)
        self.HR_LR = if_HR_LR # If the Res Block need the HR and the LR images
        self.gate = Gate_para

    def forward(self,img,HR = 0,LR = 0):

        if self.HR_LR:
            if type(img) == int and img == 0:
                img = LR * self.gate
            else:
                img = LR * self.gate + img

            HR_img = [HR,img]
            Concat = torch.cat(HR_img,dim=1)

            Q = self.conv_concat(Concat)
            Q = self.conv2(F.relu(Q))
            Q = self.conv3(Q)
            Q = self.conv4(Q)

        x = self.conv1(F.gelu(img))
        x = self.conv2(F.gelu(x))
        x = self.conv3(F.gelu(x))
        x = self.conv4(F.gelu(x))

        out = img + x

        if self.HR_LR:
            return out + Q
        else:
            # return out,x
            return out


class HR_up(nn.Module):

    def __init__(self,imgChannels=3):
        super(HR_up,self).__init__()

        self.imgChannels = imgChannels
        self.pool = nn.AvgPool2d(2,stride=2)
        self.block = Res_Block(self.imgChannels, if_HR_LR=False)

    def forward(self,img):

        block = self.block

        # The first layer of up
        x = block(img)
        out1 = block(block(x))
        out = self.pool(F.gelu(out1))

        # The second layer of up
        x = block(out)
        out2 = block(block(x))
        out = self.pool(F.gelu(out2))

        # The third layer of up
        x = block(out)
        out3 = block(block(x))
        out = self.pool(F.gelu(out3))

        # The fourth layer of up
        x = block(out)
        out4 = block(block(block(x)))

        return [out1,out2,out3,out4]

class LR_up(nn.Module):

    def __init__(self,imgChannels=3):
        super(LR_up,self).__init__()

        self.imgChannels = imgChannels
        self.pool = nn.AvgPool2d(2,stride=2)
        self.block = block = Res_Block(self.imgChannels,if_HR_LR=False)

    def forward(self,img):

        block = self.block

        # The first layer of up
        x = block(img)
        out1 = block(block(x))
        out = self.pool(F.gelu(out1))

        # The second layer of up
        x = block(out)
        out2 = block(block(x))
        out = self.pool(F.gelu(out2))

        # The third layer of up
        x = block(out)
        out3 = block(block(x))

        return [out1,out2,out3]

class SR_VAE(nn.Module):

    def __init__(self,LR_up,HR_up,imgChannels=3):
        super(SR_VAE,self).__init__()

        self.imgChannels = imgChannels
        self.lrup = LR_up
        self.hrup = HR_up
        self.pool = nn.AvgPool2d(2, stride=2)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.block = Res_Block(self.imgChannels, if_HR_LR=True)
        self.encFC1 = nn.Linear(3*32*32, 256)

    def encoder(self,HR_output,LR_output,x = 0):

        block = self.block

        x = block(x,HR_output[3],LR_output[2])
        x = block(x,HR_output[3],LR_output[2])
        x = block(x,HR_output[3],LR_output[2])
        x = block(x,HR_output[3],LR_output[2])

        return self.unpool(F.gelu(x))

    def decoder(self,input,HR_output,LR_output):

        block = self.block

        x = block(input, HR_output[2], LR_output[1])
        x = block(x, HR_output[2], LR_output[1])
        x = block(x, HR_output[2], LR_output[1])

        out = self.unpool(F.gelu(x))

        x = block(out, HR_output[1], LR_output[0])
        x = block(x, HR_output[1], LR_output[0])
        x = block(x, HR_output[1], LR_output[0])

        out = self.unpool(F.gelu(x))

        x = block(out, HR_output[0], 0)
        x = block(x, HR_output[0], 0)
        x = block(x, HR_output[0], 0)

        return F.gelu(x)

    def reparameterize(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self,img_high,img_low):

        HR_output = self.hrup(img_high)
        LR_output = self.lrup(img_low)

        output_ = self.encoder(HR_output,LR_output)
        z = self.reparameterize(output_,output_)
        output = self.decoder(z,HR_output,LR_output)

        return output,output_,output_



# class VAE(nn.Module):
#     def __init__(self, imgChannels=3, featureDim=32*24*24, zDim=256):
#         super(VAE, self).__init__()
#
#         # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
#         self.conv1 = nn.Conv2d(imgChannels, 32, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(32, imgChannels, kernel_size=3, stride=2, padding=1)
#         self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
#         self.encConv2 = nn.Conv2d(16, 32, 5)
#         self.encFC1 = nn.Linear(featureDim, zDim)
#         self.encFC2 = nn.Linear(featureDim, zDim)
#
#         # Initializing the fully-connected layer and 2 convolutional layers for decoder
#         self.decFC1 = nn.Linear(zDim, featureDim)
#         self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
#         self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)
#
#     def encoder(self, x):
#
#         # Input is fed into 2 convolutional layers sequentially
#         # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
#         # Mu and logVar are used for generating middle representation z and KL divergence loss
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = F.relu(self.encConv1(x))
#         x = F.relu(self.encConv2(x))
#         x = x.view(-1, 32*24*24)
#         mu = self.encFC1(x)
#         logVar = self.encFC2(x)
#         return mu, logVar
#
#     def reparameterize(self, mu, logVar):
#
#         #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
#         std = torch.exp(logVar/2)
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def decoder(self, z):
#
#         # z is fed back into a fully-connected layers and then into two transpose convolutional layers
#         # The generated output is the same size of the original input
#         x = F.relu(self.decFC1(z))
#         x = x.view(-1, 32, 24, 24)
#         x = F.relu(self.decConv1(x))
#         x = torch.sigmoid(self.decConv2(x))
#         return x
#
#     def forward(self, x):
#
#         # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
#         # output, mu, and logVar are returned for loss computation
#         mu, logVar = self.encoder(x)
#         z = self.reparameterize(mu, logVar)
#         out = self.decoder(z)
#         return out, mu, logVar

def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):

    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=8, sampler=RandomSampler(dataset_train), num_workers=0,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=8, num_workers=0,drop_last=True)

    return train_loader,valid_loader



if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        kernel_type='9c_meta_b3_768_512_ext_18ep',
        out_dim=4,
        data_dir='data/',
        data_folder=512,
        use_meta=False,
    )

    transforms_train, transforms_val = get_transforms(image_size=512)
    fold='0,1,2'
    folds = [int(i) for i in fold.split(',')]
    learning_rate = 1e-3
    num_epochs = 10
    LR_up = LR_up()
    HR_up = HR_up()
    #net = VAE(imgChannels=3).to(device)
    net = SR_VAE(LR_up,HR_up,imgChannels=3)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for fold in folds:
        print('fold',fold)
        if fold==0:
            train_loader, valid_loader=run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)
            for epoch in range(num_epochs):
                print('epoch',epoch)
                for idx, data in enumerate(tqdm(train_loader), 0):
                    imgs, _ = data
                    imgs = imgs.to(device)
                    imgs_low = F.interpolate(imgs, size=(128, 128), mode='bilinear', align_corners=False)
                    imgs_high = F.interpolate(imgs, size=(256,256), mode='bilinear', align_corners=False)
                    # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                    #out, mu, logVar = net(imgs,)
                    out, mu, logVar = net.forward(imgs_high,imgs_low)
                    # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                    print('kl diver',kl_divergence)

                    loss = F.binary_cross_entropy_with_logits(out, imgs_high, size_average=True) + kl_divergence

                    #loss = F.binary_cross_entropy_with_logits(out, imgs_high, size_average=True)
                    print('loss',loss)
                    # Backpropagation based on the loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('Fold{}: Epoch {}: Loss {}'.format(fold, epoch, loss))
                torch.save(net.state_dict(),f'weights/vae{epoch}_{fold}.pt')