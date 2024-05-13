import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import Resize

BATCH_SIZE = 72
# For extrapolation case, SEQ_SIZE = 4; for interpolation case, SEQ_SIZE = 2
SEQ_SIZE = 4
# SEQ_SIZE = 2
learning_rate = 0.0001
PATH_SAVE = './model/octane-fixed-extra-model.t7'
# PATH_SAVE = './model/octane-fixed-inter-model.t7'
SIDE = 512

transform_list = [
    Resize((SIDE, SIDE)),
    transforms.ToTensor()
]

data_transforms = transforms.Compose( transform_list )

def default_loader(path):
    return Image.open(path)

def to_img(x):
    x = x.view(x.shape[0], 1, SIDE, SIDE) # Reshape to (batch_size, channels, height, width)
    return x

class SeqDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqs.append(line)
        self.num_samples = len(imgseqs)
        self.imgseqs = imgseqs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        ## Here we select a sequence randomly instead of in a fixed order to shuffle the data
        current_index = np.random.choice(range(0, self.num_samples))
        imgs_path = self.imgseqs[current_index].split()
        current_imgs = []
        current_imgs_path = imgs_path[:len(imgs_path)-1]
        current_label_path = imgs_path[len(imgs_path)-1]
        current_label = self.loader(current_label_path)

        for frame in current_imgs_path:
            img = self.loader(frame)
            if self.transform is not None:
                img = self.transform(img)
            current_imgs.append(img)
        current_label = self.transform(current_label)
        batch_cur_imgs = np.stack(current_imgs, axis=0)

        return batch_cur_imgs, current_label

    def __len__(self):
        return len(self.imgseqs)


class EncoderMUG2d_LSTM(nn.Module):
    def __init__(self, input_nc=1, encode_dim=1024, lstm_hidden_size=1024, seq_len=SEQ_SIZE, num_lstm_layers=1, bidirectional=False):
        super(EncoderMUG2d_LSTM, self).__init__()
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        #1*SIDE*SIDE
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4,2,1), # 32 * 256 * 256
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1), # 64 * 128 * 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 128 * 64 * 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 256 * 32 * 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), # 512 * 16 * 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 4 * 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1), # 1024 * 1 * 1
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc = nn.Linear(1024, encode_dim) # expect every input to be a 1-D tensor of length 1024
        self.lstm = nn.LSTM(encode_dim, encode_dim, batch_first=True) # the input tensor to the LSTM should have its first dimension as the batch size

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)


    def forward(self, x):
        # x.shape [batchsize,seqsize,1,SIDE,SIDE]
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 1, SIDE, SIDE) # x.shape[batchsize * seqsize,1,SIDE,SIDE]
        # [batchsize*seqsize, 1, SIDE, SIDE] -> [batchsize * seqsize, 1024,1,1]
        x = self.encoder(x)

        #[batchsize * seqsize, 1024, 1, 1]-> [batchsize*seqsize, 1024]
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = x.view(-1, SEQ_SIZE, x.size(1))
        h0, c0 = self.init_hidden(x)
        output, (hn,cn) = self.lstm(x,(h0,c0))
        return hn


class DecoderMUG2d(nn.Module):
    def __init__(self, output_nc=1, encode_dim=1024): #output size: 512*512
        super(DecoderMUG2d, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(encode_dim, 1024*4*4),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # 512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 256*16*16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 128*32*32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64*64*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32*128*128
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16*256*256
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 16*512*512
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, output_nc, 3, stride=1, padding=1),  # 1*512*512
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.project(x) # the output of this layer is a 1D tensor (a flat tensor)
        x = x.view(-1, 1024, 4, 4) # reshape the 1D tensor into a 4D tensor with the specified dimensions
        decode = self.decoder(x)
        return decode


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.n1 = EncoderMUG2d_LSTM()
        self.n2 = DecoderMUG2d()

    def forward(self, x):
        output = self.n1(x)
        output = self.n2(output) #B*1*SIDE*SIDE
        return output

if __name__ == '__main__':
    train_data = SeqDataset(txt='./path/octane-fixed-extra-train-path.txt',transform=data_transforms)
    # train_data = SeqDataset(txt='./path/octane-fixed-inter-train-path.txt',transform=data_transforms)

    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=BATCH_SIZE, drop_last=True)

    model = net()
    if torch.cuda.is_available(): # moves the model to the GPU if one is available.
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss() # The mean squared error loss is used.

    inputs, label = next(iter(train_loader))

    print(inputs.size())
    print(label.size())

    for epoch in range(1000):
        __loss = 0
        train_loss = []
        train_acc = 0.
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

            inputs, label = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            # This line wraps the input and target data in Variable objects.
            # This is a requirement for computation with PyTorch.

            output = model(inputs)

            loss = loss_func(output, label)/label.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            __loss += loss.item()

        print('Epoch: {}, Loss: {:.6f}'.format(epoch + 1, loss.data.cpu().numpy()))

    # Save the trained LSTM model and use it to make predictions in 'predict.py'
    torch.save(model.state_dict(), PATH_SAVE)