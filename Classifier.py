import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):

    def __init__(self,in_channel, out_channel, medin_channel, stride_one = 1 ,if_connect = True, padding = 1):
        super(Resnet,self).__init__()

        self.in_channel = in_channel
        self.out_chennel = out_channel
        self.medin_channel = medin_channel
        self.stride = stride_one
        self.if_connect = if_connect
        self.padding = padding

        self.BN1 = nn.BatchNorm2d(self.in_channel,affine=True)
        self.BN2 = nn.BatchNorm2d(self.medin_channel,affine=True)
        self.conv1 = nn.Conv2d(self.in_channel,self.medin_channel,kernel_size=3,padding=self.padding,stride=self.stride)
        self.conv2 = nn.Conv2d(self.medin_channel,self.out_chennel,kernel_size=3,padding=self.padding,stride=1)

    def forward(self,img):

        x_one = F.relu(self.BN1(img))
        out_one = self.conv1(x_one)

        x_two = F.relu(self.BN2(out_one))
        out_two = self.conv2(x_two)

        if self.if_connect:
            out = img + out_two
            return out
        else:
            return out_two

class Meta_transform(nn.Module):

    def __init__(self,out_size,in_size):
        super(Meta_transform,self).__init__()

        self.in_size = in_size
        self.outsize = out_size
        self.Linear1 = nn.Linear(self.in_size,512)
        self.Linear2 = nn.Linear(512,self.outsize)

        self.Drop = nn.Dropout(0.3 ,inplace=False)
        self.BN1 = nn.BatchNorm1d(512,affine=True)
        self.BN2 = nn.BatchNorm1d(self.outsize,affine=True)

    def forward(self,meta):

        x = F.relu(self.BN1(self.Linear1(meta)))
        out_one = self.Drop(x)

        out = self.BN2(self.Linear2(out_one))
        return F.relu(out)

class SENet(nn.Module):

    def __init__(self,Resnet,in_chaanel,r_num = 16):
        super(SENet,self).__init__()

        self.r = r_num
        self.inchannel = in_chaanel
        self.resnet = Resnet

        self.Allpool = nn.AdaptiveAvgPool2d((1,1))
        self.Liear1 = nn.Linear(self.inchannel,self.r)
        self.Liear2 = nn.Linear(self.r,self.inchannel)

    def get_imgsize(self,img):
        return img.size()[2:]

    def forward(self,img):

        img_ = self.resnet(img)
        input_img = self.Allpool(img_)

        x = input_img.view(-1,self.inchannel)
        x = F.relu(self.Liear1(x))

        out = F.sigmoid(self.Liear2(x))
        weight = out.view(-1,self.inchannel,1,1)
        img_ = img_*weight

        return img_ + img

class Model_img(nn.Module):

    def __init__(self):
        super(Model_img,self).__init__()

        self.resnet1_on = Resnet(64, 64, 64)
        self.resnet1_off = Resnet(64, 64, 64, if_connect=False)

        self.resnet2_on = Resnet(128, 128, 128, stride_one=2, padding=0,if_connect=False)
        self.resnet2_off = Resnet(128, 128, 128, if_connect=False)

        self.resnet3_on = Resnet(256, 256, 256, stride_one=2, padding=0,if_connect=False)
        self.resnet3_off = Resnet(256, 256, 256, if_connect=False)

        self.resnet4_on = Resnet(128, 128, 128, stride_one=2, padding=0,if_connect=False)
        self.resnet4_off = Resnet(128, 128, 128, if_connect=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        # Attention BLock
        self.senet1 = SENet(self.resnet1_off, 64)
        self.senet2 = SENet(self.resnet2_off, 128)
        self.senet3 = SENet(self.resnet3_off, 256)

        self.conv1_ = nn.Conv2d(64,128,padding=1,stride=1,kernel_size=3)
        self.conv2 = nn.Conv2d(128,256,padding=1,stride=1,kernel_size=3)
        self.conv3 = nn.Conv2d(256,128,padding=1,stride=1,kernel_size=3)

    def forward(self,img):

        input = self.pool(self.conv1(img))

        x1 = self.resnet1_on(input)
        x1 = self.senet1(x1)
        x1 = self.conv1_(x1)

        x2 = self.resnet2_on(x1)
        x2 = self.senet2(x2)
        x2 = self.conv2(x2)

        x3 = self.resnet3_on(x2)
        x3 = self.senet3(x3)
        x3 = self.resnet3_on(x3)
        x3 = self.senet3(x3)
        x3 = self.conv3(x3)

        x4 = self.resnet4_on(x3)
        x4 = self.resnet4_off(x4)

        print(x4.size())
        return x4

class Classifier(nn.Module):

    def __init__(self,n_meta_features):
        super(Classifier,self).__init__()

        self.n_meta_feature = n_meta_features
        self.meta_transform = Meta_transform(in_size=self.n_meta_feature,out_size=6272)
        self.model = Model_img()
        self.Drop = nn.Dropout(0.5, inplace=False)
        self.Linear = nn.Linear(64*5*5,1)
        self.conv = nn.Conv2d(256,64,stride=1,kernel_size=3)

    def concat(self,img,meta):
        x = [img,meta]
        return torch.cat(x,dim=1)

    def forward(self,img,meta):

        meta = self.meta_transform(meta)
        out_img = self.model(img)

        out = self.concat(out_img,meta.view(-1,128,7,7))
        x = self.Drop(out)

        x = F.relu(self.conv(x))
        out = self.Linear(x.view(-1,64*5*5))

        return out


