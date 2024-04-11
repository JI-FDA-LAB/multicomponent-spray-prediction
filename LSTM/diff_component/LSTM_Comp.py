import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import Resize

BATCH_SIZE = 72
SEQ_SIZE = 4
learning_rate = 0.001
PATH_SAVE = './model/lstm_model.t7'
SIDE = 512
writer = SummaryWriter()
# Notice thta .t7 is an extension associated with Torch.


transform_list = [
    Resize((SIDE, SIDE)),
    transforms.ToTensor()
]

data_transforms = transforms.Compose( transform_list )

def default_loader(path):
    return Image.open(path)

def to_img(x):
    x = 0.5 * (x + 1.)  # Rescale -1~1 to 0~1
    x = x.clamp(0, 1) # Ensure values are within 0~1
    x = x.view(x.shape[0], 1, SIDE, SIDE) # Reshape to (batch_size, channels, height, width)
    # Here x.shape[0] is used to maintain the original batch size when reshaping the tensor
    # which allows the fcn to handle batches of images of any size.
    return x

def to_origin(x):
    x = x * 2 - 1
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
        # batch_cur_imgs = torch.from_numpy(np.stack(current_imgs, axis=0)).contiguous()

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
            nn.Conv2d(input_nc, 32, 4,2,1), # 32 * 128 * 128
            # This is a 2D convolutional layer. It takes an input with input_nc channels and applies 32 filters of size 4x4.
            # The stride is 2 (meaning the filters move 2 pixels at a time), and the padding is 1 (meaning the input is zero-padded by 1 pixel on each side).
            # The output of this layer will have 32 channels.
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            #32*63*63
            nn.Conv2d(32, 64, 4, 2, 1), # 64 * 64 * 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #64*31*31
            nn.Conv2d(64, 128, 4, 2, 1), # 128 * 32 * 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 256 * 16 * 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), # 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 4 * 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1), # 512 * 2 * 2
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
        #x.shape [batchsize,seqsize,1,SIDE,SIDE]
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 1, SIDE, SIDE) #x.shape[batchsize*seqsize,1,SIDE,SIDE]
        # [batchsize*seqsize, 1, SIDE, SIDE] -> [batchsize*seqsize, 1024,1,1]
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
    def __init__(self, output_nc=1, encode_dim=1024): #output size: 512*512
        super(DecoderMUG2d, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(encode_dim, 1024*4*4),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            # The size changes according to this rule: O = (I-1) * S - 2P + F ( in this case = 2*S)
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
        x = x.view(-1, 1024, 4, 4) # The line x = x.view(-1, 1024, 4, 4) reshapes the 1D tensor into a 4D tensor with the specified dimensions
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

# The if __name__ == '__main__': line is a common Python idiom.
# In Python, __name__ is a special variable. When you run a Python file directly,
# __name__ is set to '__main__'. But if you import the file as a module in another script,
# __name__ is set to the name of that file (without the .py).
if __name__ == '__main__':
    train_data = SeqDataset(txt='./training_img_path.txt',transform=data_transforms)

    # The DataLoader is a PyTorch utility for loading data in parallel.
    # It divides the data into batches of size BATCH_SIZE, shuffle the data
    # And uses 4 worker processes to load the data
    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=BATCH_SIZE, drop_last=True)

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

    # with open('loss_log.txt', 'w') as f:
    for epoch in range(1):
        __loss = 0
        print('epoch {}'.format(epoch + 1))
        train_loss = []
        train_acc = 0.
        #count = 1
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

            inputs, label = Variable(batch_x), Variable(batch_y)
            # This line wraps the input and target data in Variable objects.
            # This is a requirement for computation with PyTorch.

            output = model(inputs)

            loss = loss_func(output, label)/label.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Make the output to fit the original scaling
            # output = to_origin(output)

            __loss += loss.item()
            writer.add_scalar('Loss/train', __loss, epoch)
            print('Epoch: {}, Batch: {}, Loss: {:.6f}'.format(epoch + 1, batch_idx + 1, loss.data.cpu().numpy()))
            # Write the loss for this batch to the log file
            # f.write('Epoch: {}, Bathc: {}, Loss: {:.6f}\n'.format(epoch + 1, batch_idx + 1, loss.data.cpu().numpy()))

        print('Epoch: {}, Loss: {:.6f}'.format(epoch + 1, loss.data.cpu().numpy()))
        # f.write('Epoch: {}, Loss: {:.6f}\n'.format(epoch + 1, loss.data.cpu().numpy()))

        if (epoch + 1) % 1 == 0:  # 每 5 次，保存一下解码的图片和原图片
            pic = to_img(output.cpu().data)
            img = to_img(label.cpu().data)
            if not os.path.exists('./conv_autoencoder'):
                os.mkdir('./conv_autoencoder')
            save_image(pic, './conv_autoencoder/decode_image_{}.png'.format(epoch + 1))
            save_image(img, './conv_autoencoder/raw_image_{}.png'.format(epoch + 1))

    writer.close()
    torch.save(model.state_dict(), PATH_SAVE)

    # Load the trained model
    model.load_state_dict(torch.load(PATH_SAVE))

    # Create a DataLoader for the test set
    test_data = SeqDataset(txt='./test_img_path.txt', transform=data_transforms)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=8, batch_size=BATCH_SIZE, drop_last=True)

    # Use the model to predict
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients to save memory
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = model(inputs)
            # Here, outputs are your predictions

            # Convert the outputs and labels to images
            output_imgs = to_img(outputs.cpu().data)
            label_imgs = to_img(labels.cpu().data)

            # Save each output image and label individually
            for j in range(output_imgs.size(0)):
                output_img = output_imgs[j, ...]
                label_img = label_imgs[j, ...]

                # Create a directory to save the images if it doesn't exist
                output_dir = './test_results/output_image_{}/'.format(i + 1)
                os.makedirs(output_dir, exist_ok=True)

                # Save the output image and the corresponding label
                save_image(output_img, os.path.join(output_dir, 'output_image_{}.png'.format(j + 1)))
                save_image(label_img, os.path.join(output_dir, 'label_image_{}.png'.format(j + 1)))



            # # Create a directory to save the images, if it doesn't exist
            # if not os.path.exists('./test_results'):
            #     os.mkdir('./test_results')

            # # Save the output images and the corresponding labels
            # save_image(output_imgs, './test_results/output_image_{}.png'.format(i + 1))
            # save_image(label_imgs, './test_results/label_image_{}.png'.format(i + 1))