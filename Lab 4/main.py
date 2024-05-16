import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import sys
import torch
import numpy as np
from fractions import Fraction
import time
import cv2
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class ConvolutionNetwork(nn.Module):
    def __init__(self, kernel_size, mask, device):
        super(ConvolutionNetwork, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 1, kernel_size=kernel_size, padding='same').to(self.device)
        self.setBias()
        self.setMask(mask)

    def setBias(self):
        self.conv1.bias = nn.Parameter(torch.zeros_like(self.conv1.bias))

    def setMask(self, mask):
        mask = mask.reshape(self.conv1.weight.shape)
        self.conv1.weight = nn.Parameter(torch.tensor(mask, dtype=torch.float32).to(self.device))

    def forward(self, x):
        x = self.conv1(x)
        return x

def getKernel(kernel_path):
    kernel_array = []
    kernel_size = 0
    with open(kernel_path,'r') as file:
        kernel_size = int(file.readline().strip())

        for _ in range(kernel_size):
            line = file.readline().strip().split()
            float_array = [float(Fraction(x)) for x in line]
            kernel_array.append(float_array)
    kernel_array = np.array(kernel_array)
    flattenedArray = kernel_array.flatten()
    return (kernel_size,kernel_size), torch.tensor([flattenedArray, flattenedArray, flattenedArray])

def getting_images_batches(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset(root_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset.image_files, data_loader

def execute(images, kernel_size, kernel, device):
    conv_net = ConvolutionNetwork(kernel_size, kernel,device)
    output_images = []
    elapsed_time=0
    with torch.no_grad():
        for batch in images:
            batch = batch.to(device)
            start_time = time.time() * 1_000_000
            output = conv_net(batch)
            output = output.squeeze(1)
            numpy_images = (output.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            end_time = time.time() * 1_000_000
            elapsed_time += end_time - start_time
            output_images.extend(numpy_images)
    print("Convolution process took {:.2f} microseconds.".format(elapsed_time))
    return output_images

def writeOutput(output_images,image_names,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i,image in enumerate(output_images):
        new_image_path = os.path.join(output_dir,image_names[i].split('\\')[-1])
        cv2.imwrite(new_image_path,image)

if(len(sys.argv) == 1):
    print("Please enter input directory path")
elif(len(sys.argv) == 2):
    print("Please enter output directory path")
elif(len(sys.argv) == 3):
    print("Please enter input batch size")
elif(len(sys.argv) == 4):
    print("Please enter input kernel path")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Define input and output directories
    input_dir = sys.argv[1] 
    output_dir = sys.argv[2] 
    batch_size = int(sys.argv[3])
    kernel_path = sys.argv[4]
    
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Batch size:", batch_size)
    print("Kernel path:", kernel_path)

    image_files,images = getting_images_batches(input_dir,batch_size)
    kernel_size,kernel = getKernel(kernel_path)
    output_images = execute(images,kernel_size,kernel,device)
    writeOutput(output_images,image_files,output_dir)