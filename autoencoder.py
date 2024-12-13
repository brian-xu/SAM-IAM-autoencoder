from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader
# import matplotlib as plt
from tqdm import tqdm

image_size = 256
latent_dim = 128
input_channels = 1

def random_skew():
    datagen = transforms.Compose([transforms.Resize((image_size,image_size)),
                                  transforms.RandomAffine(degrees=0, scale=(0.85, 1.25), shear=(-45,45,-45,45)), 
                                  transforms.Lambda(lambda x: x.point(lambda y: 255 if y > 250 else 0))])
    
    return datagen

def flatten(arr):
    if len(arr.shape) == 3:
        arr = arr [:, :, 0]
    return arr/255

def get_difference_map(input_frame, output_frame):
    """ takes in two images as arrays, returns a difference map as an array """
    difference_map = output_frame - input_frame
    difference_map -= difference_map.min()
    if difference_map.max() > 0:
        return difference_map / difference_map.max()
    return difference_map


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, davis_root):
        self.image_pairs = []
        annotations = os.path.join(davis_root, "Annotations", "480p")
        for dataset in os.scandir(annotations):
            files = sorted(os.scandir(dataset.path), key= lambda x:x.path)
            files = [file for file in files if os.path.splitext(file)[1] == ".png"]
            i = 0
            j = 1
            while j < len(files):
                self.image_pairs.append([files[i], files[j]])
                i += 1
                j += 1

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_frame_path, output_frame_path = self.image_pairs[idx]
        input_frame = Image.open(input_frame_path)
        output_frame = Image.open(output_frame_path)
        diff_map = get_difference_map(flatten(np.asarray(input_frame)), flatten(np.asarray(output_frame)))
        skew = random_skew()
        return {"diff_map": diff_map, 
                "input_frame": flatten(np.asarray(skew(input_frame))), 
                "output_frame": flatten(np.asarray(skew(output_frame)))}

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super(Encoder, self).__init__()
        
        self.resize = transforms.Resize((image_size, image_size))
        self.latent_dim = latent_dim
        self.height = self.width = img_size
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten_size = (self.height // 16) * (self.width // 16)
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, self.latent_dim)

    def forward(self, diff_mask):
        diff_mask = self.resize(diff_mask)
        x = F.relu(self.conv1(diff_mask))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        latent_space = self.fc2(x)

        return latent_space

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super(Decoder, self).__init__()

        self.resize = transforms.Resize((image_size, image_size))
        self.latent_dim = latent_dim
        self.height = self.width = img_size
        self.input_channels = input_channels
        self.new_frame_size = input_channels * self.height * self.width
        
        self.input_dim = self.latent_dim*256 + self.height*self.width

        self.fc1 = nn.Linear(self.input_dim, 2048)
        self.fc2 = nn.Linear(2048, 128 * (self.height // 16) * (self.width // 16))
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

    def forward(self, latent_code, frame1):
        frame1 = self.resize(frame1)
        x = torch.hstack((latent_code.view(1, -1), frame1.view(frame1.size(0), -1)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(x.size(0), 128, self.height // 16, self.width // 16)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        
        return x

class AE(torch.nn.Module):
    def __init__(self, latent_dim, input_channels, img_size):
        super().__init__()
         
        self.encoder = Encoder(latent_dim, input_channels, img_size)
        self.decoder = Decoder(latent_dim, input_channels, img_size)

    def forward(self, diff_map, frame1):
        x = self.encoder(diff_map)
        x = self.decoder(x, frame1)

        return torch.squeeze(x, dim=0)

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = AEDataset("/workspace/DAVIS")
    train_data = DataLoader(train_set, shuffle=True)

    model = AE(latent_dim, input_channels, image_size).to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    
    epochs = 5
    outputs = []
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        i = 0
        with tqdm(train_data) as tepoch:
            for batch in tepoch:
        
                # diff_mask = diff_mask.reshape(-1, 28*28)
                
                # pass in difference mask and frame 1
                reconstructed = model(batch['diff_map'].float().to(device), batch['input_frame'].float().to(device))
                loss = loss_function(reconstructed.float().to(device), batch['output_frame'].float().to(device))
            
                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                i += 1
                optimizer.step()

                tepoch.set_postfix(loss=f"{total_loss/i:.4e}")
                # tepoch.set_postfix(recon=np.sum(reconstructed.detach().cpu().numpy()), input=np.sum(batch['input_frame'].detach().cpu().numpy()))
                
                # so we can plot if needed?
                losses.append(loss)
                if i > 5:
                    break

        # outputs.append((epochs, image, reconstructed))
    
    # # Defining the Plot Style
    # plt.style.use('fivethirtyeight')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    
    # # Plotting the last 100 values
    # plt.plot(losses[-100:])

    # save model weights
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "model_weights/autoencoder.pth"))

if __name__ == "__main__":
    train()