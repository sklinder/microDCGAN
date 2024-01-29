#!/usr/bin/env python
# coding: utf-8


'''
Author:          Steffen Klinder
Supervisor:      M.Sc. Juliane Blarr
University:      Karlsruhe Institut for Technologie  
Institute:       Institute for Applied Materials
Research Group:  Hybrid and Lightweight Materials
Group Leader:    Dr.-Ing. Wilfried Liebig

DISCLAIMER
Some sections of the following code are inspired by the "lung DCGAN 128x128" project by Milad Hasani (see https://www.kaggle.com/code/miladlink/lung-dcgan-128x128) which has been released under the Apache 2.0 open source license (https://www.apache.org/licenses/LICENSE-2.0).
The relevant sections are marked as such.
Copyright (C) 2021 Milad Hasani

Modifications copyright (C) 2023 Steffen Klinder
This modified source code is licensed under a Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)
'''

# ### Input parameters
# Set folder paths for input and output data and choose parameters for training

print('>>> Input parameters <<<')

architecture = 'DCGAN'                                           # Use architecture 'GAN' or 'DCGAN'


path_input = '../data/input/256x256/offset_x4'        # Location of real input images for training 
path_output = '../data/output/256x256/offset'                    # Location where the generated images are going to be saved

name_output = 'DCGAN_256x256_bs128_lr0001_ks4_e75_con_noweights_scans1and2_fin'

size_input = 256                                        # Pixel size of input images (need to be square for now). Currently no resize operations in this code...
batch_size = 128                                          # Number of training images in every batch. Total number of training batches loaded = (Number of input images / batch_size)
num_epochs = 75                                       # Number of training epochs (=complete passes through the training dataset)
lr = 0.0001

format_input = '.jpg'                                   # Format of input images
format_output = '.jpg'                                  # Format of input images

betas = (0.5, 0.999)                                      # Adam-Optimizer coefficients for computing running averages of gradient and its square, the pyTorch default is (0.9, 0.999)

### DCGAN ###
ks=4
apply_weights = False
#############

number_gpu = 4                                          # Number of GPUs used device 'cuda' is available, needs to be > 0 to work on GPU!
#number_workers = 2                                      # Number of workers for dataloader

random_seed = 123                                       # Set seed for random operations (shuffle, ...)

#resize_input = 128                                      # If size_input != resize_input the image will be resized using transforms.resize

font_size = 22


# Import all necessary packages

print('>>> Imports <<<')

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils


from torchinfo import summary                                       # Displays a TensorFlow like summary of the initialised model; 'pip install torchinfo'

from glob import glob
import numpy as np
import signal
import sys
import datetime
from datetime import datetime

import csv
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings     # needed to supress some insignificant warnings

from torchmetrics.image.fid import FrechetInceptionDistance         
from torchmetrics import StructuralSimilarityIndexMeasure


print(f'\n{datetime.now().replace(microsecond=0)}\n')


plt.ioff()  # turn interactive plotting of 

try:
  mpl.use('agg')
except Exception as e:
  print(e)
  

# Get command line arguments for job execution

number_gpu = 4

try:
  job_partition=sys.argv[1]
  job_ntasks=sys.argv[2]
  job_time=sys.argv[3]
  job_mem=sys.argv[4]
  job_gres=sys.argv[5]
  job_ID=sys.argv[6]
  

  print('\n'+'=== JOB PARAMETERS =======================================================================')
  print(f'Partition:     {job_partition}')
  print(f'Tasks:         {job_ntasks}')
  print(f'Time-limit:    {job_time}')
  print(f'Memory (RAM):  {job_mem}')
  print(f'GPUs:          {number_gpu}')
  print(f'Job-ID:        {job_ID}')
  print('==========================================================================================\n\n')
except Exception as e:
  print(e)


import gc
torch.cuda.empty_cache()
gc.collect()


# ### Utilities
# Define helper functions

# ##### Text and Print

print('>>> Utilities and helper functions <<<')

def print_write(text, file):
    '''
    Function to print text to console output and write to a specified file simultaneously.
    To be used inside 
    
        with open(path, mode, encoding="utf-8") as file:
            print_write(text, file)
    
    Parameters
    ----------
    text: str
        Contains the information
    file: str
        Name of opened file
    '''
    print(text)
    file.write(text + '\n')


# ##### CSV

def to_csv(path, list):
    '''
    Function to save array in a new .csv file.

    Parameters
    ----------
    path: str
        Name of File in current working directory or path to file in different directory
    list: list
        List containing the data
    '''
    with open(path, 'w') as file:
        write = csv.writer(file)
        write.writerow(list)


# ### Initialization
# Initialise GPU (Nvidia cuda) if available to later use for computation instead of CPU (works only if cuda framework was set up previously):
# ##### CUDA 

device = ("cuda" if torch.cuda.is_available() and number_gpu > 0 else "cpu")
print(f"Running computation on device: {device}\n")

#print(torch.get_num_threads())
#torch.set_num_interop_threads()    # Inter-op parallelism for cpu
#torch.set_num_threads(10)           # Intra-op parallelism for cpu ???


# ##### Random seed 

torch.manual_seed(random_seed);


# ##### LaTex 

# In[9]:


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': font_size})


# ### Load and process input

# Set function to normalize input data and transform to pyTorch tensor:

# ##### Transformation

# Create a list of transformations
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#transforms = transforms.Compose([transforms.Resize(resize_input), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # resize operation is outsourced
print('Apply transformations:\n\n' + str(transforms))


# Load the real image dataset from input folder:

print(f'\n>>> Loading real dataset {datetime.now().replace(microsecond=0)} <<<')

train_set = []
label = 0           

number_images = len(glob(f'{path_input}/*{format_input}'))
print(f'\nLoading {number_images} images\n')

for file in (glob(f'{path_input}/*{format_input}')):                   # Get training data from specified folder
    #pbar.set_description('Loading training images from directory')
    img = Image.open(file)                      # Use PIL to load images
    img_tensor = transforms(img)                # Apply the transformations defined above, 'torch.Size([1, size_input, size_input])' with only one color channel (gray image)
    train_set.append((img_tensor, label))       # Append tupel ( , )
    

# Distribute images from the training dataset into batches with size 'batch_size':


# Divide the train set into multiple shuffled batches
# 'drop_last=True' ensures that no partially filled batches are generated (if number of images != integer multiple of batch size, the remaining images are ignored)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)

# see pin_memory=True and num_workers=<int>



print(f'Total number of batches from DataLoader: {len(train_loader)}\n')

# Use 'pass', 'plt.show()' or ';' or bind to a variable '_ = ' to suppress unwanted info about the plot 
# plt.show()


# ### Evaluation
# ##### Frechet Inception Distance 

print(f'>>> (Try to) initialize FID {datetime.now().replace(microsecond=0)} <<<')

fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, device=device)
  #fid = FrechetInceptionDistance(normalize=True, feature=64, reset_real_features=False, device=device)                  ### changed from ipynb-file !!!
  # an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following: 64, 192, 768, 2048
  
  # Update the pretrained model with real validation data, only once necessary because of 'reset_real_features=False' 
try:
  for batch in (train_loader):
      #pbar.set_description('Training FID network on real batches')
      batch = (batch[0].repeat(1,3,1,1))*255
      fid.update(batch.type(torch.uint8), real=True)     # batch[0] contains the images whereas batch[1] contains the labels       
except Exception as e:
  print(e)              

def FID(fake):
    '''
    Function to calculate the FID for a fake image with pretrained inceptionv3 network
    
    Parameters
    ----------
    fake: np array of size [1, 1, width, height]
        Generated (=fake) numpy image array to compare with training data
    
    Returns
    ----------
    value: float
        Computed FID score
    '''
    fid.reset()                                          # reset all fake features (reset_real_features=False) 
    fake = (fake.repeat(1,3,1,1))*255
    fid.update(fake.type(torch.uint8), real=False)
    return fid.compute().item()

# ##### Euklidean Distance

print(f'\n>>> Euklidean distance {datetime.now().replace(microsecond=0)} <<<')

def euklidean_distance(tensor_a, tensor_b):
    '''
    Calculates the euklidean distance of two torch tensors of the same size.

    Parameters
    ----------
    tensor_a: torch tensor
        Image A (e.g. fake)
    tensor_b: torch tensor
        Image B (e.g. real)
    
    Returns
    ----------
    value: float
        Euklidean distance
    '''
    return torch.sqrt(((tensor_a - tensor_b) ** 2).sum())


# ##### Structural Similarity Index Measure 

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

### Model

### GENERATOR ###

###############################################################################
# GAN 
###############################################################################
if architecture == 'GAN':

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 64*64),
                
                nn.ReLU(),
                nn.Linear(64*64, 128*128),
                
                nn.ReLU(),
                nn.Linear(128*128, 180*180),
                
                # nn.ReLU(),
                # nn.Linear(128*128, 256*256),

                # nn.Linear(100, 128),            
                # nn.BatchNorm1d(128, 0.8),
                # nn.LeakyReLU(0.2, inplace=True),

                # nn.Linear(128, 16*16),            
                # nn.BatchNorm1d(16*16, 0.8),
                # nn.LeakyReLU(0.2, inplace=True),

                # nn.Linear(16*16, 64*64),            
                # nn.BatchNorm1d(64*64, 0.8),
                # nn.LeakyReLU(0.2, inplace=True),

                # nn.Linear(64*64, 128*128),            
                # nn.BatchNorm1d(128*128, 0.8),
                # nn.LeakyReLU(0.2, inplace=True),

                # nn.Linear(128*128, 256*256),            
                # nn.BatchNorm1d(256*256, 0.8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Tanh(),                      # Tanh ensures that output is in [-1, 1]
            )

        def forward(self, input):
            output = self.model(input)
            output = output.view(input.size(0), 1, size_input, size_input)        # reshape to pyTorch tensor (unflatten)
            return output
    
    generator = Generator(number_gpu).to(device=device)

    with open(path_output+'/'+name_output+'_generator.log', 'w', encoding="utf-8") as file:         # Save architecture summary in .txt file
        print_write(str(generator), file)
        ###
        warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')       # Otherwise a warning message for summary command will be shown
        ###
        print_write('\n'+str(summary(generator, input_size = (batch_size, 100))), file)

###############################################################################
# DCGAN 
###############################################################################
elif architecture == 'DCGAN':
    

    '''
    The following section is inspired by code from Milad Hasani.
 
    Copyright (C) 2021 Milad Hasani
    Modifications copyright (C) 2023 Steffen Klinder
    '''
    
    # Fixed network parameters
    features_g = 64     # Size of feature maps in generator
    stride=2            # Stride of kernel

    class Generator (nn.Module):
        
        def __init__ (self, ngpu, features_g):
            super (Generator, self).__init__ ()
            self.ngpu = ngpu
            # Input: N x z_dim x 1 x 1
            self.model = nn.Sequential (
                self._block (100, features_g * 64, 4, stride, 0),                               # 4x4, maybe use bias=False (like in official example)
                self._block (features_g * 64, features_g * 32, ks, stride, 1),                      # 8x8
                self._block (features_g * 32, features_g * 16, ks, stride, 1),                      # 8x8
                self._block (features_g * 16, features_g * 8, ks, stride, 1),                       # 16x16
                self._block (features_g * 8, features_g * 4, ks, stride, 1),                       # 32x32
                self._block (features_g * 4, features_g * 2, ks, stride, 1),                       # 64x64
                nn.ConvTranspose2d (features_g * 2, 1, kernel_size = ks, stride = stride, padding = 1, bias = False), # 128x128
                nn.Tanh ()
            )
            
        def _block (self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential (
                nn.ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d (out_channels),
                nn.ReLU ()
            )
        
        def forward (self, input):
            #input = input.unsqueeze(-1).unsqueeze(-1)       # reshape noise vector from (batch_size, 1) to (batch_size, 1, 1, 1)
            return self.model(input)

    '''
    
    '''
    
    generator = Generator(number_gpu, features_g).to(device=device)

    with open(path_output+'/'+name_output+'_generator.log', 'w', encoding="utf-8") as file:         # Save architecture summary in .txt file
        
        try:
            print('\n>>> Generator <<<\n')
            file.write(f'{datetime.now().replace(microsecond=0)}\n\n')
            print_write(str(generator), file)
            ###
            warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')       # Otherwise a warning message for summary command will be shown
            ###
            print_write('\n'+str(summary(generator, input_size = (batch_size, 100, 1, 1))), file)
        except Exception as e:
            print(e)

###############################################################################
            
# Handle multi-GPU if desired
if (device == 'cuda') and (number_gpu > 1):
    print(f'\nRunning generator on {number_gpu} GPUs\n')
    generator = nn.DataParallel(generator, list(range(number_gpu)))


### DISCRIMINATOR ###

###############################################################################
# GAN 
###############################################################################
if architecture == 'GAN':

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.model = nn.Sequential(
                #nn.Linear(512*512, 256*256),
                #nn.ReLU(),
                #nn.Dropout(0.3),

                nn.Linear(180*180, 128*128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128*128, 64*64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64*64, 32*32),
                nn.ReLU(),
                nn.Linear(32*32, 16*16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16*16, 1),
                

                # nn.Linear(256*256, 128*128),            # use np.prod(img_shape)
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(128*128, 64*64),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(64*64, 16*16),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(16*16, 1),
                nn.Sigmoid(),               # Sigmoid ensures that output is in [0, 1]
            )

        def forward(self, input):
            input = input.view(input.size(0), size_input*size_input)      # reshape pyTorch tensor (flatten), flatten is simply a convenient alias of a common use-case of view
            return self.model(input)   
    
    discriminator = Discriminator(number_gpu).to(device=device)

###############################################################################
# DCGAN 
###############################################################################
elif architecture == 'DCGAN':
    

    # Fixed network parameters
    features_d = 64     # Size of feature maps in generator
    img_channels = 1    # number of channels
    stride=2            # Stride of kernel

    class Discriminator (nn.Module):
   
        def __init__ (self, ngpu, img_channels, features_d):
            super (Discriminator, self).__init__ ()
            self.ngpu = ngpu
            # Input: N x channels_img x 128 x 128
            self.model = nn.Sequential (
                nn.Conv2d (img_channels, features_d, kernel_size = ks, stride = stride, padding = 1, bias = False), # 64x64
                nn.LeakyReLU (0.2),
                self._block (features_d * 1, features_d * 2, ks, stride, 1),                          # 32x32
                self._block (features_d * 2, features_d * 4, ks, stride, 1),                          # 16x16
                self._block (features_d * 4, features_d * 8, ks, stride, 1),                          # 8x8
                self._block (features_d * 8, features_d * 16, ks, stride, 1),                         # 4x4
                self._block (features_d * 16, features_d * 32, ks, stride, 1),                         # 4x4
                nn.Conv2d (features_d * 32, 1, kernel_size = 4, stride = 1, padding = 0),        # 1x1
                nn.Sigmoid ()
            )

        
        def _block (self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential (
                nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d (out_channels),
                nn.LeakyReLU (0.2)
            )
        
        def forward (self, input):
            #input = input.view(-1)      # reshape pyTorch tensor (flatten), flatten is simply a convenient alias of a common use-case of view
            return self.model(input).reshape(int(batch_size/number_gpu), 1)
        
        
    discriminator = Discriminator(number_gpu, img_channels, features_d).to(device=device)

###############################################################################

# Handle multi-GPU if desired
if (device == 'cuda') and (number_gpu > 1):
    print(f'\nRunning generator on {number_gpu} GPUs\n')
    discriminator = nn.DataParallel(discriminator, list(range(number_gpu)))

try:
    with open(path_output+'/'+name_output+'_discriminator.log', 'w', encoding="utf-8") as file:     # Save architecture summary in .txt file
        
        print('\n>>> Discriminator <<<\n')
        file.write(f'{datetime.now().replace(microsecond=0)}\n\n')
        print_write(str(discriminator), file)
        sys.stdout.flush()
        ###
        warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')       # Otherwise a warning message for summary command will be shown
        ###
        print_write('\n'+str(summary(discriminator, input_size=(batch_size, 1, size_input, size_input))), file)
        sys.stdout.flush()
except Exception as e:
    print(e)



# #### Weights Initialization

print('\n>>> Initialize Weights <<<')

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if apply_weights and architecture!='GAN':
    generator.apply(weights_init)
    discriminator.apply(weights_init)


# #### Loss function

print('>>> Define loss function <<<')

loss_function = nn.BCELoss()        # Binary Cross Entropy


# #### Adam Optimizer 

print('>>> Define optimizers <<<')

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)


# #### Noise

print('>>> Define noise vectors <<<')

def noise(batch_size, size):
    '''
    Function that generates noise samples (=latent space) as generator input: a 1D vector of gaussian sampled random values 
    
    Parameters
    ----------
    batch_size: int
        Size of batches from dataloader
    size: int
        Width/height of noise input (square images, width==height, e.g. 10x10)
    
    Returns
    ----------
    noise: torch tensor
        Noise vector of size [batch_size, 1, size, size]
    '''
    if architecture == 'GAN':
        return torch.randn((batch_size, size)).to(device=device)    # torch.Size([batch_size, 100])
    elif architecture == 'DCGAN':
        return torch.randn((batch_size, size, 1, 1)).to(device=device)    # torch.Size([batch_size, 100, 1, 1])
    elif architecture == 'LDCGAN':
        return torch.randn((batch_size, size, 1, 1)).to(device=device)    # torch.Size([batch_size, 100, 1, 1])
    
fixed_noise = noise(batch_size, 100)


# ### Training
# Create empty lists to save progress during training

iterations = 0
epochs = 0

######################

img_list = []       # Save generated image batch (not already grid due to computation time) of every (few) training rounds to show animation/video later
G_losses = []       # Save generator and discriminator loss for plot options       --> change plot to start at 1 instead of adding value 0
D_losses = []
D_losses_real = []
D_losses_fake = []
FID_list = []

c_lambda = 10

score_real = []
score_fake = []
score_disc = []
#img_generated = []

live_progress = False    # Disable for (much!) faster calculation, enable for live update of image and loss function plot
intervall = 50          # Intervall for live progress and animation (for after computation)


def job_summary(status, t_start, t_end):
    with open(path_output+'/'+name_output+'_job_summary.log', 'w') as file:
        print('\n\n')
        print_write('=== COMPUTATION SUMMARY ==================================================================', file)
        print(f'Status:        {status}')
        file.write(f'Status:        {status}\n')
        print_write(f'Start:         {t_start.replace(microsecond=0)}', file)
        print_write(f'End:           {t_end.replace(microsecond=0)}', file)
        print_write(f'Duration:      {str(t_end-t_start).split(".")[0]}', file)
        print_write('------------------------------------------------------------------------------------------', file)
        print_write(f'Architecture:  {architecture}', file)
        print_write(f'Epochs:        {epochs}', file)
        print_write(f'Batches/Epoch: {len(train_loader)}', file)
        print_write(f'Batch size:    {batch_size}', file)
        print_write(f'Iterations:    {iterations}', file)
        print_write(f'Input images:  {number_images}', file)
        print_write('==========================================================================================\n\n', file)


print('\n\n'+'=== COMPUTATION PARAMETERS ===============================================================')
print(f'Architecture:  {architecture}')
print(f'Input images:  {number_images}')
print(f'Batch size:    {batch_size}')
print(f'Epochs:        {num_epochs}')
print(f'Batches/Epoch: {len(train_loader)}')
print(f'Learn rate:    {lr}')
print(f'GPUs:          {number_gpu}')
print('==========================================================================================')

print('\n>>> Start of training <<<\n')

# Initialize plots
if live_progress:       # Make sure live progress is of for running this file
    
    fig,ax = plt.subplots(2,1, figsize=(15, 5))  

    # Display generated image examples of last iteration
    ax[0].set_title(f'Generated Samples (live updated every epoch)')
    ax[0].axis('off')
    ax[0].imshow(np.transpose(vutils.make_grid(torch.ones((8, 1, size_input, size_input)), padding=5, normalize=True, nrow=8, pad_value=1.0).cpu(),(1,2,0)))     # Update the live image with newly generated samples
    
    # Display plot of ganerator and discriminatro loss functions
    ax[1].set_title('Generator and Discriminator Loss (live updated every epoch)')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Iterations')

    hdisplay = display.display('', display_id=True)
    hdisplay.update(fig)

    def live_plot():
        ax[0].imshow(vutils.make_grid(fixed_generated_samples.to(device)[:8], padding=5, normalize=True, nrow=8, pad_value=1.0).cpu().permute(1,2,0))     # Update the live image with newly generated samples
        #hdisplay.update(fig1)

        ax[1].plot(G_losses, label="Generator" if epoch == 0 else "", color="green")        # use if statement to not show duplicates at each iteration
        ax[1].plot(D_losses, label="Discriminator" if epoch == 0 else "", color="blue")     # marker="o"

        ax[1].legend(loc="upper left")
        
        hdisplay.update(fig)
           
def signal_handler(signal, frame):      
    print(f'\n>>> Job aborted after {iterations} iterations <<<')
    time_stop = datetime.now()
    job_summary('manually aborted', time_start, time_stop)
    sys.exit('Cell execution manually stopped')
    

signal.signal(signal.SIGINT, signal_handler)        # Catch error if execution is interupted

##################################################################################################################
### Main training routine 
##################################################################################################################

time_start = datetime.now()            

for epoch in range(num_epochs):

    print(f'Epoch: {str(epoch+1).zfill(len(str(num_epochs)))}/{num_epochs}, Time: {datetime.now().replace(microsecond=0)}')
    
    for n, (real_samples, _ ) in enumerate(train_loader):  
        
        # Data for training the discriminator

        real_samples = real_samples.to(device=device)       # Send to device
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

        # Generate noise vector
        latent_space_samples = noise(batch_size, 100)
        generated_samples = generator(latent_space_samples)                  
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)

        ############################
        # Update DISCRIMINATOR 
        ###########################

        discriminator.zero_grad()

        discriminator_output_real = discriminator(real_samples)
        loss_disc_real = loss_function(discriminator_output_real, real_samples_labels)
        discriminator_output_fake = discriminator(generated_samples)
        loss_disc_fake = loss_function(discriminator_output_fake, generated_samples_labels)

        loss_discriminator = (loss_disc_real+loss_disc_fake)/2
        loss_discriminator.backward()       # call backward and than update the discriminator bias
        optimizer_discriminator.step()
        
        # Data for training the generator
        latent_space_samples = noise(batch_size, 100)

        ############################
        # Update GENERATOR 
        ###########################
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Save losses of this round into array
        G_losses.append(loss_generator.item())
        D_losses.append(loss_discriminator.item())
        D_losses_real.append(loss_disc_real.item())
        D_losses_fake.append(loss_disc_fake.item()) 

        iterations += 1     # Keep track of the number of iterations and epochs
    
    fixed_generated_samples = generator(fixed_noise)
    img_list.append(fixed_generated_samples.detach().cpu())
    try:
      FID_list.append(FID(generated_samples.detach().cpu()))       # Calcute FID once after every epoche
    except Exception as e:
      print(e)
    
    if live_progress:
        live_plot() 

    epochs +=1

time_stop = datetime.now()

print('\n>>> End of training <<<\n')
job_summary('completed', time_start, time_stop)


# if live_progress:        
#     plt.close(fig)        


from torchvision.utils import save_image


for i, image in enumerate(generated_samples):
    save_image(image, f'img{i}.png', normalize=True)

to_csv(path_output+'/'+name_output+'_loss_generator.csv', G_losses)
to_csv(path_output+'/'+name_output+'_loss_discriminator.csv', D_losses)
to_csv(path_output+'/'+name_output+'_loss_disc_real.csv', D_losses_real)
to_csv(path_output+'/'+name_output+'_loss_disc_fake.csv', D_losses_fake)
to_csv(path_output+'/'+name_output+'_FID.csv', FID_list)


# ### Plots

try:
  fig, ax1 = plt.subplots(figsize=(15, 5))
  ax1.plot(range(1, len(D_losses_real)+1, 1), D_losses_real, label="Loss real", color="blue")  
  ax1.plot(range(1, len(D_losses_fake)+1, 1), D_losses_fake, label="Loss fake", color="red")        
  ax1.set_title('Discriminator Loss')
  ax1.legend(loc="upper right")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")
 
  plt.savefig(path_output+'/'+name_output+'_loss_real_fake.png', dpi=250, bbox_inches='tight')
  plt.savefig(path_output+'/'+name_output+'_loss_real_fake.eps', format='eps', bbox_inches='tight')

except Exception as e:
  print(e)


# Loss functions

try:
  fig, ax1 = plt.subplots(figsize=(15, 5))
  ax1.plot(range(1, len(G_losses)+1, 1), G_losses, label="Generator" , color="green")        
  ax1.plot(range(1, len(D_losses)+1, 1), D_losses, label="Discriminator", color="blue")     # markplt.legend(loc="upper right")
  ax1.set_title('Generator and Discriminator Loss')
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Loss")
  ax1.legend(loc="upper right")
  ax2 = ax1.twiny()                       # Add second axis
  ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
  ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
  ax2.spines['bottom'].set_position(('outward', 70))
  ax2.set_xlabel('Epochs')
  ax2.set_xlim(ax1.get_xlim())
  
  x_ticks = np.arange(0, epochs+1, 5)
  ax2.set_xticks(x_ticks*len(train_loader))
  ax2.set_xticklabels(x_ticks)
  
  props = dict(boxstyle='round', facecolor = 'wheat', alpha = 0.5)
  plt.text(.02, .93, f'batch size: {batch_size}\nlearn rate: {lr}', fontsize=22, transform = ax1.transAxes, verticalalignment='top', bbox=props)
  
  #plt.plot()
  plt.savefig(path_output+'/'+name_output+'_losses.png', dpi=250, bbox_inches='tight')
  plt.savefig(path_output+'/'+name_output+'_losses.eps', format='eps', bbox_inches='tight')

except Exception as e:
  print(e)


# Frechet Inception Distance

try:
  fig, ax1 = plt.subplots(figsize=(15, 5))
  ax1.plot(range(1, len(FID_list)+1, 1), FID_list, label="FID" , color="black")        
  ax1.set_title('Frechet Inception Distance')
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("FID")
  #ax1.legend(loc="upper right")
  
  plt.savefig(path_output+'/'+name_output+'_FID.png', dpi=250, bbox_inches='tight')
  plt.savefig(path_output+'/'+name_output+'_FID.eps', format='eps', bbox_inches='tight')

except Exception as e:
  print(e)


# Show last batch of generated images

# plt.figure(figsize=(30, 30))
# plt.axis("off")
# plt.title("Generated Images (fake, last batch)")
# plt.imshow(vutils.make_grid(generated_samples.to(device), padding=5,  normalize=True, nrow=8, pad_value=1.0).cpu().permute(1,2,0)) 


# In comparison to real input images

# real_batch = next(iter(train_loader))
# plt.figure(figsize=(30, 30))
# plt.axis("off")
# plt.title("Training Images (ground truth, example batch)")
# plt.imshow(vutils.make_grid(real_batch[0].to(device)[:], padding=5, normalize=True, nrow=8, pad_value=1.0).cpu().permute(1,2,0))
# plt.show()

try:
  real_batch = next(iter(train_loader))
  
  fig, ax = plt.subplots(1, 2, figsize=(15, 30))
  k = 32
  ax[0].imshow(vutils.make_grid(generated_samples.to(device)[:k], padding=5,  normalize=True, nrow=4, pad_value=1.0).cpu().permute(1,2,0))
  ax[0].set_title('Generated Images (fake, last batch)')
  ax[0].axis('off')
  
  ax[1].imshow(vutils.make_grid(real_batch[0].to(device)[:k], padding=5, normalize=True, nrow=4, pad_value=1.0).cpu().permute(1,2,0))
  ax[1].set_title('Training Images (ground truth, example batch)')
  ax[1].axis('off')
  
  plt.tight_layout()
  #plt.show()
  plt.savefig(path_output+'/'+name_output+'_comparison.png', dpi=250, bbox_inches='tight')
  plt.savefig(path_output+'/'+name_output+'_comparison.eps', format='eps', bbox_inches='tight')
  #plt.show()
  
except Exception as e:
  print(e)


# ### Nearest Neighbors

# ##### Euklidean Distance

print('>>> Calculating nearest neighbor (ED) <<<')
print(f'\n{datetime.now().replace(microsecond=0)}\n')

try:

  # returns enumerate with index of k nearest neighbours in training set and the corresponding distance value
  def get_k_nearest_samples(fake_image, training_images, k=4):
      '''
      Searches for the k-nearest samples in the dataset of a given image based on the euclidean distance.
      '''
      dist = [euklidean_distance(fake_image, real_image) for real_image in training_images]
      
      return np.argsort(dist)[:k], np.sort(dist)[:k] # argsort returns the indices that sort the array; index, distance value
      
      
  # Get necessary images
  train_imgs = [x[0] for x in train_set]     # get array of real images without corresponding label
  generated_samples = generated_samples.cpu().detach()
  
  
  
  eukl_dist = [(get_k_nearest_samples(generated_samples[i], train_imgs, k=1)[1])[0] for i in range(batch_size)]
  to_csv(path_output+'/'+name_output+'_NearestNeighbor_ED.csv', eukl_dist)
  
  try:
    np.savetxt(path_output+'/'+name_output+'NN_distribution.csv', np.sort(eukl_dist), delimiter=",")
  except:
    pass
  
  ### Plot euklidean distance for all images samples in generated batch
  fig, ax1 = plt.subplots(figsize=(30, 10))
  ax1.plot(np.arange(1, batch_size+1, 1), np.sort(eukl_dist), color="darkorange")        # use if statement to not show duplicates at each iteration
  ax1.set_title('Minimum euklidean distance of all images in one batch (sorted)')
  ax1.set_xlabel("Image number")
  ax1.set_ylabel("Euklidean Distance")
  
  plt.axhline(y=np.mean(eukl_dist), color='black', linestyle='--', label= 'Mean euklidean distance: %.4f (Standard deviation: %.4f)' % (np.mean(eukl_dist), np.std(eukl_dist)))
  ax1.legend(loc="upper right")
  #ax1.set_xticks(np.arange(1, batch_size+1, 1.0))
  #ax1.set_xlim(1, 24)
  ax1.grid()
  to_csv(path_output+'/'+name_output+'_NN.csv', np.sort(eukl_dist))
  plt.savefig(path_output+'/'+name_output+'_nearest_Neighbour.png', dpi=250, bbox_inches='tight')
  
  # print('==========================================================================================')
  # print('Mean euklidean distance:', np.mean(eukl_dist))
  # print('Standard deviation:', np.std(eukl_dist))
  # print('==========================================================================================')
  
  #ax1.legend(loc="upper right")
  
  
  k = 4               # number of nearest neighbours to be displayed
  eukl_index = [(get_k_nearest_samples(generated_samples[i], train_imgs, k=k)[0]) for i in range(k+1)]
  
  ### Plot nearest neighbours
  for i in range (4):
      f, ax = plt.subplots(1,k+1, figsize=(30, 10))
      ax[0].imshow(generated_samples[i].reshape(size_input, size_input), cmap="gray")
      ax[0].set_yticks([])
      ax[0].set_xticks([])
      ax[0].set_title("Generated")
      #for axis in ['top', 'bottom', 'left', 'right']:
      #    ax[0].spines[axis].set_linewidth(2.5)  # change width
      #    ax[0].spines[axis].set_color('darkorange')    # change color
      
      for j in range(k):
          ax[j+1].imshow(train_imgs[(eukl_index[i])[j]].reshape(size_input, size_input), cmap="gray")  # display only 0,4,8 because otherwise the first images are the same, just augmented (flipped)
          ax[j+1].axis('off')
          ax[j+1].set_title(f"Nearest Neighbour {j+1}")
          # ax[2].imshow(samples[x[1]].reshape(size_input, size_input), cmap="gray")
          # ax[2].axis('off')
          # ax[2].set_title("Nearest Neighbour 2")
          # ax[3].imshow(samples[x[2]].reshape(size_input, size_input), cmap="gray")
          # ax[3].axis('off')
          # ax[3].set_title("Nearest Neighbour 3")
      plt.savefig(path_output+'/'+name_output+f'_nearest_neighbour_{i}.png', dpi=250, bbox_inches='tight')
      plt.savefig(path_output+'/'+name_output+f'_nearest_neighbour_{i}.eps', format='eps', bbox_inches='tight')

except Exception as e:
  print(e)

# ##### SSIM ## Disabled do to high computation time

# Searches for the k-nearest samples in the dataset of a given image based on the euclidean distance.
'''
try:
  # returns enumerate with index of k nearest neighbours in training set and the corresponding distance value
  def get_k_nearest_samples(fake_image, training_images, k=4):
      
      
      
      dist = [ssim(fake_image.view(1,1,size_input,size_input), real_image.view(1,1,size_input,size_input)) for real_image in training_images]
      
      return np.argsort(dist)[::-1][:k], np.sort(dist)[::-1][:k] # argsort returns the indices that sort the array; index, distance value
      
  # Get necessary images
  train_imgs = [x[0] for x in train_set]     # get array of real images without corresponding label
  generated_samples = generated_samples.cpu().detach()
  
  
  
  eukl_dist = [(get_k_nearest_samples(generated_samples[i], train_imgs, k=1)[1])[0] for i in range(batch_size)]
  to_csv(path_output+'/'+name_output+'_NearestNeighbor_ED.csv', eukl_dist)
  
  
  ### Plot euklidean distance for all images samples in generated batch
  fig, ax1 = plt.subplots(figsize=(30, 10))
  ax1.plot(np.arange(1, batch_size+1, 1), np.sort(eukl_dist), color="darkorange")        # use if statement to not show duplicates at each iteration
  ax1.set_title('Minimum euklidean distance of all images in one batch (sorted)')
  ax1.set_xlabel("Image number")
  ax1.set_ylabel("Euklidean Distance")
  
  plt.axhline(y=np.mean(eukl_dist), color='black', linestyle='--', label= 'Mean euklidean distance: %.4f (Standard deviation: %.4f)' % (np.mean(eukl_dist), np.std(eukl_dist)))
  ax1.legend(loc="upper right")
  #ax1.set_xticks(np.arange(1, batch_size+1, 1.0))
  #ax1.set_xlim(1, 24)
  ax1.grid()
  to_csv(path_output+'/'+name_output+'_NN_SSIM.csv', np.sort(eukl_dist))
  plt.savefig(path_output+'/'+name_output+'_nearest_Neighbour_SSIM.png', dpi=250)
  
  print('==========================================================================================')
  print('Mean euklidean distance:', np.mean(eukl_dist))
  print('Standard deviation:', np.std(eukl_dist))
  print('==========================================================================================')
  
  ax1.legend(loc="upper right")
  
  
  k = 4               # number of nearest neighbours to be displayed
  eukl_index = [(get_k_nearest_samples(generated_samples[i], train_imgs, k=k)[0]) for i in range(k+1)]
  
  ### Plot nearest neighbours
  for i in range (4):
      f, ax = plt.subplots(1,k+1, figsize=(30, 10))
      ax[0].imshow(generated_samples[i].reshape(size_input, size_input), cmap="gray")
      ax[0].set_yticks([])
      ax[0].set_xticks([])
      ax[0].set_title("Generated")
      for axis in ['top', 'bottom', 'left', 'right']:
          ax[0].spines[axis].set_linewidth(2.5)  # change width
          ax[0].spines[axis].set_color('darkorange')    # change color
      
      for j in range(k):
          ax[j+1].imshow(train_imgs[(eukl_index[i])[j]].reshape(size_input, size_input), cmap="gray")  # display only 0,4,8 because otherwise the first images are the same, just augmented (flipped)
          ax[j+1].axis('off')
          ax[j+1].set_title(f"Nearest Neighbour {j+1}")
          # ax[2].imshow(samples[x[1]].reshape(size_input, size_input), cmap="gray")
          # ax[2].axis('off')
          # ax[2].set_title("Nearest Neighbour 2")
          # ax[3].imshow(samples[x[2]].reshape(size_input, size_input), cmap="gray")
          # ax[3].axis('off')
          # ax[3].set_title("Nearest Neighbour 3")
      plt.savefig(path_output+'/'+name_output+f'_nearest_neighbour_SSIM_{i}.png', dpi=250)
      plt.savefig(path_output+'/'+name_output+f'_nearest_neighbour_SSIM_{i}.eps', format='eps')

except Exception as e:
  print(e)
'''

# #### Save evolution progress visualization (.mp4 video)

# In[ ]:

print(f'>>> Saving video {datetime.now().replace(microsecond=0)} <<<')
#print(f'\n{datetime.now().replace(microsecond=0)}\n')

n_img = 32      # Do not show full batch if batch_size is to large

###################################
# Set the text and font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (0,0,0)
thickness = 1
###################################

print(f'Shape of img_list: {len(img_list)}')

frames = [(vutils.make_grid(frame[:n_img], padding=4, normalize=True, nrow=8, pad_value=1.0)*255) for frame in img_list]    # Generate grids of images from list of image batches


frames = torch.stack(frames, dim=0).permute(0, 2,3,1) # Stack gray (1 channel) to rgb (3 channels) and change order

#Set video parameters
fps = 10                                    # frames per second
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # format
fourcc = cv2.VideoWriter_fourcc(*'DIVX')    # format
video_size = (frames[0].size(1), frames[0].size(0)+20)                    # size of video (must fit size of frames)
#video_writer = cv2.VideoWriter(path_output+'/'+name_output + '_evolution.mp4', fourcc, fps, video_size)
video_writer = cv2.VideoWriter(path_output+'/'+name_output + '_evolution.avi', fourcc, fps, video_size)


for iteration, frame in enumerate(frames, 1):
    
    #pbar.set_description("Saving video")

    frame = frame.numpy().astype(np.uint8)                      # Convert to numpy array and then to uint8 format (converts float values to int)
    frame = np.ascontiguousarray(frame, dtype=np.uint8)         # Seems to be necessary for some reason... (elements are stored in adjacent memory locations that follow a specific order)
    #img = cv2.resize(img, video_size)
    
    text = name_output + ' | '+ str(time_start.replace(microsecond=0)) + ' | Batch Size ' + str(batch_size) + f" | Epoch {iteration}"                  # Specify text that will be displayed at the top of video frame
    frame = cv2.copyMakeBorder(frame, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))    # add white border at the top to write on
    cv2.putText(frame, text, (5, 15), font, font_scale, color, thickness)
    video_writer.write(frame)
    
    try:
      cv2.imwrite(path_output+'/'+f'E_{str(iteration).zfill(len(str(len(img_list))))}_{lr}.jpg', frame)
    except Exception as e:
      print(e)
      cv2.imwrite(f'E{iteration}.jpg', frame)

video_writer.release()

print(f'>>> Finished {datetime.now().replace(microsecond=0)} <<<')