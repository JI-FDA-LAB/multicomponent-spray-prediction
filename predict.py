import os
import torch
from torchvision.utils import save_image
from LSTM_Comp import net, SeqDataset, to_img, Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
import multiprocessing

PATH_SAVE = './model/lstm_model.t7'
SIDE = 512
BATCH_SIZE = 72

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Load the trained model
    model = net()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(PATH_SAVE))
        model.cuda()
    else:
        model.load_state_dict(torch.load(PATH_SAVE, map_location=torch.device('cpu')))

    transform_list = [
        Resize((SIDE, SIDE)),
        transforms.ToTensor()
    ]

    data_transforms = transforms.Compose( transform_list )
    # Create a DataLoader for the test set
    test_data = SeqDataset(txt='./test_img_path.txt', transform=data_transforms)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=8, batch_size=BATCH_SIZE, drop_last=True)

    # Use the model to predict
    model.eval()
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