# -*- coding: utf-8 -*-
"""
pytorch: 1.1.0
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
# from torchsummary import summary

from dataset.RCS import RCS_data


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
# lr 要夠小才不會讓g_loss有變大趨勢(可能Discriminator已經overfit)
lr = 0.00001
z_dim = 64
image_length = 3
image_dim = image_length * image_length * 1
batch_size = 25
num_epochs = 90
epoch_record = 30


disc = Discriminator(image_dim).to(device)
# summary(disc,(25,9))
gen = Generator(z_dim, image_dim).to(device)
# summary(gen,(25,64))
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# 歸一化
transforms = transforms.Compose(
    [transforms.ToTensor()]
)



dataset = RCS_data.RCS_data_from_excel("./dataset/RCS/train_freq_5_ship.xlsx")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

step = 0
D_losses = []
G_losses = []
D_probs = []

for epoch in range(1,num_epochs+1):
    for batch_idx, (real ,label) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator
        output = disc(fake).view(-1)
        
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            D_probs.append(output[14])
            G_losses.append(lossG)
            D_losses.append(lossD)
            
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f} D_prob: {output[14]:.4f}"
            )
            
            with torch.no_grad():
            
                fake = gen(noise).reshape(-1, 1, 9, 1)
                fake_img = torch.abs(fake).cpu().numpy()

                data = real.reshape(-1, 1, 9, 1)
                data_img = data.cpu().numpy()
                
                for i in range(len(data)):
                    fake_img[i] = fake_img[i] * (torch.max(data[i]).cpu().numpy())
    
                sorted_label = sorted(label)
                sorted_index = sorted(range(len(label)),key= lambda k:label[k])
                
                if epoch % epoch_record == 0:
                    csv_title = f'output_epoch_{epoch}.csv'
                    with open(csv_title,"w", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        fake_list = []
                        for i in range(fake_img.shape[0]):
                            fake_one_label = list(str(sorted_label[i].cpu().numpy()))
                            fake_one_img = list(fake_img[sorted_index[i]].reshape(9))
                            fake_combine = fake_one_label + fake_one_img
                            fake_list.append(fake_combine)
                        
                        fake_list_rows = zip(*fake_list)
                        writer.writerows(fake_list_rows)
                    
                    # fig, axs = plt.subplots(5, 1)
                    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    # wspace=None, hspace=1)
                    # for img, lbl, ax in zip(data_img, label, axs.ravel()):
                    #     ax.imshow(img, cmap='gray')
                    #     ax.axis('off')
                    #     ax.set_title(f'ship_type #{lbl}')
                    # plt.suptitle("Real Images", fontsize=15,color = "b", y= 1)
                    # plt.savefig(f'./G/real_img_{epoch}.png')
                        
                    # fig, axs = plt.subplots(5, 1)
                    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    # wspace=None, hspace=1)
                    # for img, lbl, ax in zip(fake_img, label, axs.ravel()):
                    #     ax.imshow(img, cmap='gray')
                    #     ax.axis('off')
                    #     ax.set_title(f'ship_type #{lbl}')
                    # plt.suptitle("Fake Images", fontsize=15,color = "g", y= 1)
                    # plt.savefig(f'./G/fake_img_{epoch}.png')
                    # plt.show()
                step += 1

plt.figure()
plt.title("Discriminator Loss During Training",y=1.05, fontsize=15)
plt.plot(range(len(D_losses)),D_losses,'b-')
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.savefig('D_loss.png')

plt.figure()
plt.title("Generator Loss During Training",y=1.05, fontsize=15)
plt.plot(range(len(G_losses)),G_losses,'g-')
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.savefig('G_loss.png')

plt.figure()
plt.title("The Discriminator's Estimate Of The Probability That A Fake Data Is Real",y=1.05, fontsize=15)
plt.plot(range(len(D_probs)),D_probs,'k-')
plt.ylim(0.4,0.6)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Probability", fontsize=15)
plt.savefig('D_score.png')
plt.show()

import pandas as pd
df = pd.DataFrame({'epoch':range(len(D_losses)),
                   'D_loss': [i.item() for i in D_losses],
                   'G_loss': [i.item() for i in G_losses],
                   'D_score': [i.item() for i in D_probs]})
df.to_csv('result.csv',index=False)
