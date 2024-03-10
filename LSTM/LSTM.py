## This is for LSTM 

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

BATCH_SIZE = 10
SEQ_SIZE = 15
learning_rate = 0.01
PATH_SAVE = './model/lstm_model.t7'
# Notice thta .t7 is an extension associated with Torch.

transform_list = [
    Resize((1024, 1024)),
    transforms.ToTensor()
]

data_transforms = transforms.Compose( transform_list )

def default_loader(path):
    return Image.open(path)

def to_img(x):
    x = 0.5 * (x + 1.)  # Rescale -1~1 to 0~1
    x = x.clamp(0, 1) # Ensure values are within 0~1
    x = x.view(x.shape[0], 1, 1024, 1024) # Reshape to (batch_size, channels, height, width)
    # Here x.shape[0] is used to maintain the original batch size when reshaping the tensor
    # which allows the fcn to handle batches of images of any size.
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
        ## Here we select a sequence randomly instead of in a fixed order
        ## to shuffle the data, which is a common practice in ML.
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
        #print(current_label.shape)
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
        #1*1024*1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4,2,1), # 32 * 512 * 512
            # This is a 2D convolutional layer. It takes an input with input_nc channels and applies 32 filters of size 4x4.
            # The stride is 2 (meaning the filters move 2 pixels at a time), and the padding is 1 (meaning the input is zero-padded by 1 pixel on each side).
            # The output of this layer will have 32 channels.
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            #32*63*63
            nn.Conv2d(32, 64, 4, 2, 1), # 64 * 256 * 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #64*31*31
            nn.Conv2d(64, 128, 4, 2, 1), # 128 * 128 * 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 256 * 64 * 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), # 512 * 32 * 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 16 * 16
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

        self.fc = nn.Linear(1024, encode_dim)
        self.lstm = nn.LSTM(encode_dim, encode_dim, batch_first=True)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)


    def forward(self, x):
        #x.shape [batchsize,seqsize,1,1024,1024]
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 1, 1024, 1024) #x.shape[batchsize*seqsize,1,1024,1024]
        # [batchsize*seqsize, 1, 1024, 1024] -> [batchsize*seqsize, 1024,1,1]
        x = self.encoder(x)

        ## The rest of forward fcn does not depend on the image size
        #[batchsize * seqsize, 1024, 1, 1]-> [batchsize*seqsize, 1024]
        x = x.view(-1, 1024)
        # [batchsize * seqsize, 1024]
        x = self.fc(x)
        # [batchsize , seqsize ,1024]
        x = x.view(-1, SEQ_SIZE, x.size(1))
        # -1 here is a placeholder. PyTorch will automatically fill in based on the size of
        # the original tensor and the other dimensions specified.
        h0, c0 = self.init_hidden(x)
        output, (hn,cn) = self.lstm(x,(h0,c0))
        return hn


class DecoderMUG2d(nn.Module):
    def __init__(self, output_nc=1, encode_dim=1024): #output size: 64x64
        super(DecoderMUG2d, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(encode_dim, 1024*1*1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # 512*2*2
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 256*4*4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 128*8*8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64*16*16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32*32*32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16*64*64
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 8*128*128
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),  # 4*256*256
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1),  # 2*512*512
            nn.BatchNorm2d(2),
            nn.ReLU(True),

            nn.ConvTranspose2d(2, output_nc, 4, stride=2, padding=1),  # 1*1024*1024
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.project(x)
        x = x.view(-1, 1024, 1, 1)
        decode = self.decoder(x)
        return decode


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.n1 = EncoderMUG2d_LSTM()
        self.n2 = DecoderMUG2d()

    def forward(self, x):
        output = self.n1(x)
        output = self.n2(output) #B*1*1024*1024
        return output

# The if __name__ == '__main__': line is a common Python idiom.
# In Python, __name__ is a special variable. When you run a Python file directly,
# __name__ is set to '__main__'. But if you import the file as a module in another script,
# __name__ is set to the name of that file (without the .py).
if __name__ == '__main__':
    train_data = SeqDataset(txt='./img_path.txt',transform=data_transforms)

    # The DataLoader is a PyTorch utility for loading data in parallel.
    # It divides the data into batches of size BATCH_SIZE, shuffle the data
    # And uses 4 worker processes to load the data
    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=BATCH_SIZE)

    model = net()
    if torch.cuda.is_available(): # moves the model to the GPU if one is available.
        # This allows the model to be trained faster
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss() # The mean squared error loss is used.

    inputs, label = next(iter(train_loader))
    # This line gets the first batch of data from the train_loader
    print(inputs.size())
    print(label.size())

    for epoch in range(10):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        train_acc = 0.
        #count = 1
        for batch_x, batch_y in train_loader:

            inputs, label = Variable(batch_x), Variable(batch_y)
            # This line wraps the input and target data in Variable objects.
            # This is a requirement for computation with PyTorch.

            output = model(inputs)
            print(output.size())
            print(inputs.size())
            print(label.size())
            loss = loss_func(output, label)/label.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data.cpu().numpy()))

        if (epoch + 1) % 5 == 0:  # 每 5 次，保存一下解码的图片和原图片
            pic = to_img(output.cpu().data)
            img = to_img(label.cpu().data)
            if not os.path.exists('./conv_autoencoder'):
                os.mkdir('./conv_autoencoder')
            save_image(pic, './conv_autoencoder/decode_image_{}.png'.format(epoch + 1))
            save_image(img, './conv_autoencoder/raw_image_{}.png'.format(epoch + 1))
        #count = count +1

    torch.save(model.state_dict(), PATH_SAVE)