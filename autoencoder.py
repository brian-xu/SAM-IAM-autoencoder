from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import v2
import torchvision
from os import listdir
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib as plt
from tqdm import tqdm

def skew(image):
    datagen = v2.Compose([
        v2.RandomResizedCrop(size=480, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32)])
    
    skewed_image = datagen(image)
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(skewed_image)

def get_difference_map(image1, image2):
    """ takes in two images as arrays, returns a difference map as an array """
    difference_map = torch.abs(image1 - image2)
    return difference_map

def load_data():
    train_diff_maps = []
    skewed_frame0 = []
    test_images = []
    folder_dir = "bear"
    for i, image in enumerate(listdir(folder_dir)[:len(folder_dir)-1]):
        img0 = np.asarray(Image.open(f"{folder_dir}/{image}"))
        img1_path = listdir(folder_dir)[i+1]
        img1 = np.asarray(Image.open(f"{folder_dir}/{img1_path}"))
        # img_to_tensor = torchvision.transforms.ToImage()
        diff_map = get_difference_map(torch.Tensor(img0), torch.Tensor(img1))
        train_diff_maps.append(diff_map)
        skewed_img0 = skew(torch.unsqueeze(torch.Tensor(img0), 0))
        skewed_frame0.append(skewed_img0)

    for image in listdir(folder_dir)[1:]: 
        if image == "00077.png":
            continue
        img = np.asarray(Image.open(f"{folder_dir}/{image}"))
        skewed_img = skew(torch.unsqueeze(torch.Tensor(img), 0))
        test_images.append(np.asarray(skewed_img))
    
    return train_diff_maps, skewed_frame0, test_images

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, train_diff_maps, skewed_imgs, test_imgs):
        self.train_diff_maps = train_diff_maps
        self.skewed_imgs = skewed_imgs
        self.test_imgs = test_imgs

    def __len__(self):
        return len(self.train_diff_maps)

    def __getitem__(self, idx):
        return {"diff_map": self.train_diff_maps[idx], 
                "skewed_frame1": self.skewed_imgs[idx], 
                "test_img": self.test_imgs[idx]}

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.height, self.width = img_size
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten_size = 256 * (self.height // 16) * (self.width // 16)
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, self.latent_dim)

    def forward(self, diff_mask):
        x = self.conv1(diff_mask)
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        latent_space = self.fc2(x)

        return latent_space

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.height, self.width = img_size
        self.input_channels = input_channels
        self.new_frame_size = input_channels * self.height * self.width
        
        self.fc1 = nn.Linear(self.latent_dim + self.new_frame_size, 2048)
        self.fc2 = nn.Linear(2048, 128 * (self.height // 16) * (self.width // 16))
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, latent_code, frame1):
        x = torch.cat((latent_code, frame1.view(frame1.size(0), -1)), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, self.height // 16, self.width // 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x

class AE(torch.nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super().__init__()
         
        self.encoder = Encoder(latent_dim, input_channels, img_size)
        self.decoder = Decoder(latent_dim, input_channels, img_size)

    def forward(self, diff_map, frame1):
        x = self.encoder(diff_map)
        x = self.decoder(x, frame1)
        return torch.nn.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

def main():

    train_diff_maps, skewed_frame1, test_images = load_data()
    train_set = AEDataset(train_diff_maps, skewed_frame1, test_images)
    train_data = DataLoader(train_set)

    dims = train_diff_maps[0].shape[-2:]
    latent_dim = 128
    if len(train_diff_maps[0].shape) == 2:
        input_channels = 1
    else:
        input_channels = 3

    model = AE(latent_dim, input_channels, dims)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    
    epochs = 5
    outputs = []
    losses = []

    progress_bar = tqdm(range(len(train_data)))
    for epoch in range(epochs):
        for batch in train_data:
        
            # diff_mask = diff_mask.reshape(-1, 28*28)
            
            # pass in difference mask and frame 1
            reconstructed = model(batch['diff_map'], batch['skewed_frame1'])
            
            loss = loss_function(reconstructed, batch['test_img'])
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # so we can plot if needed?
            losses.append(loss)

            progress_bar.update(1)
        # outputs.append((epochs, image, reconstructed))
    
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-100:])

main()