"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: FER2013 Visualization with matplotlib & tensorboard
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse 
from dataset import FER2013
from utils import get_label_emotion

import torch.utils.tensorboard as tensorboard


class FER2013_Visualizer:
    def __init__(self, n_grid = 3):
        self.n_grid = n_grid

    def init_fig_axes(self):
        self.fig, self.axes = plt.subplots(nrows=self.n_grid, ncols=self.n_grid, figsize=(8,8), dpi=100)

    def visualize(self, images, emotions):
        self.init_fig_axes()
        n_images = images.shape[0]
        assert n_images == self.n_grid * self.n_grid
                
        i_data = 0
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                self.axes[i,j].imshow(images[i_data], cmap='gray')
                emotion = get_label_emotion(emotions[i_data])
                self.axes[i,j].set_title(emotion)
                i_data += 1
                

    def show(self):
        self.fig.tight_layout()
        print('Exit figure to continue plotting\n')
        plt.show(self.fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='train', help='dataset mode')
    parser.add_argument('--grid_size',type=int,choices=[2,3,4,5],default=3,help='size of matplotlib vis grid of images')
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--tensorboard', action='store_true', help='tensorboard visualization')
    parser.add_argument('--logdir', type=str, default='checkpoint/tensorboard', help='tensorboard logdir')
    parser.add_argument('--stop', type=int, default=5, help='number of batches to be visualized in tensorboard')
    parser.add_argument('--batch_size', type=int, default=64,help='num of images in each tensorboard batch vis')
    args = parser.parse_args()

    dataset = FER2013(root=args.datapath, mode = args.mode)

    # Visualization
    if not args.tensorboard:
        dataloader = DataLoader(dataset, batch_size=args.grid_size * args.grid_size, shuffle=False)
        visualizer = FER2013_Visualizer(n_grid=args.grid_size)

        for images, emotions in dataloader:
            visualizer.visualize(images.numpy(), emotions.numpy())
            visualizer.show()

    # Tensorboard
    else:
        writer = tensorboard.SummaryWriter(args.logdir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        batch = 0
        for images, emotions in dataloader:
            batch += 1

            # add 1 in channels dim => (batch_size, 48, 48, 1)
            images = torch.unsqueeze(images, axis=3) 
            writer.add_images("images", images, global_step=batch, dataformats="NHWC")
            print ("*" * 60, f'\n\n\t Saved {args.batch_size} images with Step{batch}. run tensorboard @ project root')
            
            if batch == args.stop:
                break

        writer.close()

