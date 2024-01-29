'''
Author:          Steffen Klinder
Supervisor:      M.Sc. Juliane Blarr
University:      Karlsruhe Institut for Technologie  
Institute:       Institute for Applied Materials
Research Group:  Hybrid and Lightweight Materials
Group Leader:    Dr.-Ing. Wilfried Liebig

Last modified:   2023-07-12

DISCLAIMER
Some sections of the following code are inspired by the "lung DCGAN 128x128" project by Milad Hasani (see https://www.kaggle.com/code/miladlink/lung-dcgan-128x128) which has been released under the Apache 2.0 open source license (https://www.apache.org/licenses/LICENSE-2.0).
The relevant sections are marked as such.
Copyright (C) 2021 Milad Hasani

Modifications copyright (C) 2023 Steffen Klinder
This modified source code is licensed under a Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)
'''


# This file contains the network architectures (models). Available Models: GAN, DCGAN


import torch.nn as nn                # Docs: https://pytorch.org/docs/stable/nn.html  


### GAN #################################################################################################

class GAN_Generator(nn.Module):
    
    def __init__(self):
        super(GAN_Generator, self).__init__()
                
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32),
            nn.ReLU(),
            nn.Linear(32*32, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 128*128),
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

            nn.Tanh(),                      # Tanh activation ensures that output is in [-1, 1]
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 1, 128, 128)        # reshape to pyTorch tensor (unflatten)
        return x
    

class GAN_Discriminator(nn.Module):
    def __init__(self):
        super(GAN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            #nn.Linear(512*512, 256*256),
            #nn.ReLU(),
            #nn.Dropout(0.3),

            # nn.Linear(256*256, 128*128),
            # nn.ReLU(),
            # nn.Dropout(0.3),
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
            nn.Sigmoid(),               # Sigmoid activation ensures that output is in [0, 1]
        )

    def forward(self, x):
        x = x.view(x.size(0), 128*128)      # reshape pyTorch tensor (flatten), flatten is simply a convenient alias of a common use-case of view
        return self.model(x)   
    
    
 # see https://www.kaggle.com/code/miladlink/lung-dcgan-128x128       
### DCGAN ###############################################################################################
'''
The following section is inspired by code from Milad Hasani.
Copyright (C) 2021 Milad Hasani
Modifications copyright (C) 2023 Steffen Klinder
'''
class DCGAN_Generator(nn.Module):

    def __init__ (self, img_channels, num_features, stride=2, kernel_size=4):
        super (DCGAN_Generator, self).__init__ ()
                
        # Input: N x z_dim x 1 x 1
        self.model = nn.Sequential (
            self._block (100, num_features * 32, 4, stride, 0),                               # 4x4, maybe use bias=False (like in official example)
            self._block (num_features * 32, num_features * 16, kernel_size, stride, 1),                      # 8x8
            self._block (num_features * 16, num_features * 8, kernel_size, stride, 1),                       # 16x16
            self._block (num_features * 8, num_features * 4, kernel_size, stride, 1),                       # 32x32
            self._block (num_features * 4, num_features * 2, kernel_size, stride, 1),                       # 64x64
            nn.ConvTranspose2d (num_features * 2, img_channels, kernel_size = kernel_size, stride = stride, padding = 1, bias = False), # 128x128
            nn.Tanh ()         # Tanh activation ensures that output is in [-1, 1]
        )

    def _block (self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential (
            nn.ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU ()
        )

    def forward (self, x):
        #input = input.unsqueeze(-1).unsqueeze(-1)       # reshape noise vector from (batch_size, 1) to (batch_size, 1, 1, 1)
        return self.model(x)
    
    
class DCGAN_Discriminator (nn.Module):
   
        def __init__ (self, img_channels, num_features, stride=2, kernel_size=4):
            super (DCGAN_Discriminator, self).__init__ ()

            # Input: N x channels_img x 128 x 128
            self.model = nn.Sequential (
                nn.Conv2d (img_channels, num_features, kernel_size = kernel_size, stride = stride, padding = 1, bias = False), # 64x64
                nn.LeakyReLU (0.2),
                self._block (num_features * 1, num_features * 2, kernel_size, stride, 1),                          # 32x32
                self._block (num_features * 2, num_features * 4, kernel_size, stride, 1),                          # 16x16
                self._block (num_features * 4, num_features * 8, kernel_size, stride, 1),                          # 8x8
                self._block (num_features * 8, num_features * 16, kernel_size, stride, 1),                         # 4x4
                nn.Conv2d (num_features * 16, 1, kernel_size = 4, stride = stride, padding = 0),        # 1x1
                nn.Sigmoid ()         # Sigmoid activation ensures that output is in [0, 1]
            )
        
        def _block (self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential (
                nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d (out_channels),
                nn.LeakyReLU (0.2)
            )
        
        def forward (self, x):
            #input = input.view(-1)      # reshape pyTorch tensor (flatten), flatten is simply a convenient alias of a common use-case of view
            #return self.model(input).view(batch_size, 1)
            x = self.model(x).reshape(128, 1)
            return x


