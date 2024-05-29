import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

class ImagePreprocess:
    def __init__(self, path, scale=25):
        self.im = imread(path)
        self.M, self.N, self.O = self.im.shape
        self.seg_map = np.zeros((self.M, self.N), dtype=int)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.scale = scale
        self.im_copy = np.copy(self.im)
        self.final_image = np.copy(self.im)
    
    def in_bounds(self, x, y):
        return 0 <= x < self.M and 0 <= y < self.N
    
    def region_growing(self):
        segment_id = 1
        total_iterations = self.M * self.N
        with tqdm(total=total_iterations, desc='Segmenting') as pbar:
            for i in range(self.M):
                for j in range(self.N):
                    if self.seg_map[i, j] == 0:
                        queue = deque([(i, j)])
                        self.seg_map[i, j] = segment_id
                        segment_color = self.im[i, j].astype(np.float64)
                        segment_size = 1
                        
                        while queue:
                            x, y = queue.popleft()
                            for dx, dy in self.directions:
                                nx, ny = x + dx, y + dy
                                if self.in_bounds(nx, ny) and self.seg_map[nx, ny] == 0:
                                    color_diff = np.linalg.norm(self.im[nx, ny] - segment_color / segment_size)
                                    if color_diff < self.scale:
                                        queue.append((nx, ny))
                                        self.seg_map[nx, ny] = segment_id
                                        segment_color += self.im[nx, ny]
                                        segment_size += 1
                            pbar.update(1)
                        segment_id += 1
        
        # Update image with segment average colors by iterating over every element
        segment_sums = np.zeros((segment_id, self.O), dtype=np.float64)
        segment_counts = np.zeros(segment_id, dtype=int)
        
        with tqdm(total=total_iterations, desc='Calculating Averages') as pbar:
            for i in range(self.M):
                for j in range(self.N):
                    seg_id = self.seg_map[i, j]
                    segment_sums[seg_id] += self.im[i, j]
                    segment_counts[seg_id] += 1
                    pbar.update(1)
        
        valid_counts = segment_counts.copy()
        valid_counts[valid_counts == 0] = 1 
        segment_averages = segment_sums / valid_counts[:, None]

        default_color = np.mean(self.im, axis=(0, 1))
        segment_averages[segment_counts == 0] = default_color
        
        with tqdm(total=total_iterations, desc='Updating Segments') as pbar:
            for i in range(self.M):
                for j in range(self.N):
                    seg_id = self.seg_map[i, j]
                    self.im_copy[i, j] = segment_averages[seg_id]
                    pbar.update(1)
    
    def draw_boundaries(self):
        boundary_mask = np.zeros((self.M, self.N), dtype=bool)
        with tqdm(total=self.M * self.N, desc='Drawing Boundaries') as pbar:
            for i in range(1, self.M-1):
                for j in range(1, self.N-1):
                    if (self.seg_map[i, j] != self.seg_map[i+1, j] or
                        self.seg_map[i, j] != self.seg_map[i, j+1] or
                        self.seg_map[i, j] != self.seg_map[i-1, j] or
                        self.seg_map[i, j] != self.seg_map[i, j-1]):
                        boundary_mask[i, j] = True
                    pbar.update(1)
                    
        self.final_image[boundary_mask] = [0, 0, 0]
    
    def plot_images(self):
        self.region_growing()
        self.draw_boundaries()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.im)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(self.final_image)
        plt.title('Segmented Image with Boundaries')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.im_copy)
        plt.title('Segmented Image without Boundaries')
        plt.axis('off')
        
        plt.show()