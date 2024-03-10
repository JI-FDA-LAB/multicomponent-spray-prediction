from PIL import Image
import torch
import torchvision.transforms as transforms

# Define the transformation
transform = transforms.Compose([
    # transforms.Grayscale(),  # Convert image to grayscale
    transforms.ToTensor()    # Convert image to tensor
])

# Open the image
image = Image.open('/Users/rhine_e/Downloads/CrossPatternData/LSTM/Data/001/00100001.png')

# Apply the transformation
tensor = transform(image)

print(tensor)
print(tensor.size(0))
print(torch.ones(76,76))
print(torch.ones(3,3).size())