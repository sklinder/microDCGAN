'''
Author:          Steffen Klinder
Supervisor:      M.Sc. Juliane Blarr
University:      Karlsruhe Institut for Technologie  
Institute:       Institute for Applied Materials
Research Group:  Hybrid and Lightweight Materials
Group Leader:    Dr.-Ing. Wilfried Liebig

Last modified:   2023-07-14
'''

# This code opens a file dialog to let you select multiple image files from a folder.
# The selected images are then loaded and can be saved as a single video file.
# Make sure the images (video frames) are therefore named in ascending alphabetical order,
# e.g. img001.jpg, img002.jpg, img003.jpg, ... with img001.jpg beeing the first video frame and so on
# (and not img1.jpg, img2.jpg, ... as this would get mixed up with img10.jpg, img11.jpg, ... !)


import os
from tkinter import Tk, filedialog
from PIL import Image, ImageTk

import numpy as np
import cv2
from tqdm import tqdm


### Set video parameters #################################################################
name = 'evolution'                          # Placeholder base name that will be shown when choosing the output directory and filename
format = '*.mp4'                            # Video format, has to match video codec (fourcc)!
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Video codec FourCC (Four Character Code, e.g. "DIVX", "XVID", "MJPG", "X264", "WMV1", "WMV2", "FMP4", "mp4v", "avc1", "I420", "IYUV", "mpg1", "H264", ...)
fps = 10                                    # Frames per second
#########################################################################################


# Set window icon (KIT logo, optional)
window = Tk()
window.withdraw()


print('Select video frames. All images must be of the same format and size and numbered alphabetically in ascending order.')

# Get the path of the image files and sort them in ascending order
filenames = sorted(filedialog.askopenfilenames(
    filetypes=[("Image Files", ".jpg .jpeg .png.")], title='Select image files'))

# Exit program early if no image is selected
if filenames == []:
    print('\nNo images selected. Quitting program.')
    quit()

print(f'Opened {len(filenames)} images.')

# Get the path of the image directory
path_output = os.path.dirname(filenames[0])

# Open first image to determine image specs
frame = cv2.imread(filenames[0])
height, width, layers = frame.shape

print(f'\nWidth:  {width}px\nHeight: {height}px')

# The size of the video must equal the size of the frames
size = (width, height)

# Get the path of the video output directory
video_path = filedialog.asksaveasfilename(initialfile=name, title='Save as', filetypes = [('Video files', format), ('All Files', '*.*')], defaultextension=format)

# Exit program early if no output path is selected
if video_path == '':
    print('\nNo output path and filename selected. Quitting program.')
    quit()

# Initialize the video writer
video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)

# Open images and write them to the video as single frames
for filename in tqdm(filenames, desc=f'Saving video'):
    frame = cv2.imread(filename)
    video_writer.write(frame)

# Close the video writer
video_writer.release()

print(f'Saved video to {path_output}')
