'''
Pre-processing with pretrained Unet
Image:
Step1: Boundary crop and insert the new image size (optinal)
Step2: Shrink the image to required input size
Step3: Get the segmentation masks
Step4: Zoom in the segmentation masks
Step5: Get a squared bbox 
    Get the center point C of the lung mask bbox
    Get the shortest distance from C to boundaries
Step6: Crop the image with the squared bbox
Step7 Shrink the image to 512*512

Note: should set determinitic
'''
import torch
import time
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import argparse
from src.models import PretrainedUNet

class MIMIC(Dataset):
    def __init__(self, 
                 root_dir,
                 resize512, resize256,  
                 new_root_512, new_root_256,
                 csv_path, 
                 use_frontal=True,
                 ):
        self.root_dir = root_dir
        self.resize512 = resize512
        self.resize256 = resize256
        self.new_root_512 = new_root_512
        self.new_root_256 = new_root_256

        # load data from csv
        self.df = pd.read_csv(csv_path)# len=377110
        if use_frontal:
            # filtering by ViewPosition
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])] # len=243334
        self._num_images = len(self.df)

    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):
          row = self.df.iloc[idx]
          image_path = f'{self.root_dir}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
          
          if self.resize512:
            dst_dir_512 =  f'{self.new_root_512}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}'
            dst_path_512 = f'{dst_dir_512}/{row["dicom_id"]}.jpg'
            if not os.path.exists(dst_dir_512):
                    os.makedirs(dst_dir_512)
          else:
              dst_path_512 = ""
          
          if self.resize256:
            dst_dir_256 =  f'{self.new_root_256}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}'
            dst_path_256 = f'{dst_dir_256}/{row["dicom_id"]}.jpg'
            if not os.path.exists(dst_dir_256):
                    os.makedirs(dst_dir_256)
          else:
              dst_path_256 = ""
          
          print('')
          print(f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg')

          # Read the original chest X-ray image
          origin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # np.array, dtype=uint8
          # shrinking input image, [cv2.INTER_AREA for shrinking and cv2.INTER_LINEAR for zooming]
          input0 = cv2.resize(origin, dsize=(512, 512), interpolation=cv2.INTER_AREA) # np.array, uint8
          # ToTensor().normalize to [0., 1.]->[-1, 1]
          input = transforms.ToTensor()(input0) - 0.5 # PyTorch tensor [1, 512, 512], torch.float32

          return origin, input, dst_path_512, dst_path_256

def main(args):
    start_time = time.time()
    resize512=True; resize256=False
    root_cropped_dir = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/files"
    root_512_dir = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/files"
    root_256_dir = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files"
    df_copped_path = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/mimic-cxr-2.0.0-metadata-cropped.csv"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_6v = PretrainedUNet(in_channels=1, out_channels=2,  batch_norm=True, upscale_mode="bilinear")
    unet_6v.load_state_dict(torch.load(f'.src/trained_models/unet-6v.pt', map_location=torch.device("cpu")))
    unet_6v.to(device)
    unet_6v.eval()

    dataloader = DataLoader(MIMIC(root_dir = root_cropped_dir,
                            resize512=resize512, resize256=resize256,     
                            new_root_512=root_512_dir, new_root_256=root_256_dir,
                            csv_path = df_copped_path,
                            use_frontal=True
                            ), 
                batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=True)

    for idx, (origin, input, dst_path_512, dst_path_256) in enumerate(dataloader):
        dst_path_512 = dst_path_512[0]
        dst_path_256 = dst_path_256[0]
        origin = origin[0] # CHW->HW, torch.uint8
        input = input.to(device)
        with torch.no_grad():
            # Mask from unet-6v
            mask_6v = unet_6v(input) # 12HW, torch.float32
            softmax_6v = torch.nn.functional.log_softmax(mask_6v, dim=1) # 12HW, torch.float32
            mask = torch.argmax(softmax_6v, dim=1).float() # 1HW, torch.int64->float32

        mask = mask[0].cpu().detach().numpy()# HW, np.array, float32
        (row, col) = origin.shape
        if np.all(mask == 0): # Perform center crop with shorter side
            print(f'Zero mask!')
            # Get the center point
            mid_row = int(row / 2)
            mid_col = int(col / 2)
            # Get the min side length
            crop_size = int(np.min([row, col])/2)
        else: 
            ########################Filter FPs in Mask########################
            # Use connected component labeling from scipy.ndimage
            labeled_mask, num_labels = ndimage.label(mask)
            if num_labels>2:
                print(f'Labels: {num_labels}')
                # Calculate the size of each connected component
                sizes = ndimage.sum(mask, labeled_mask, range(num_labels + 1))
                # Find the indices of the two largest components
                largest_indices = np.argsort(sizes)[-2:]
                # Create an empty mask for the two largest components
                mask_array = np.zeros_like(mask)
                # Keep the two largest components
                for index in largest_indices:
                    mask_array[labeled_mask == index] = 1 # np.array [0,1], uint8
            else:
                mask_array = mask.copy()
            ########################Zooming Mask########################
            mask_resize = cv2.resize(mask_array, dsize=(col, row), interpolation=cv2.INTER_NEAREST) # np.array([0., 1.], float32)
            # Check size
            assert origin.shape[0]== mask_resize.shape[0] and origin.shape[1]== mask_resize.shape[1], 'Orign and Mask size mismatch!'

            ########################Square Crop########################
            # Find coordinates of ones pixels (lung region)
            ones_coordinates = np.argwhere(mask_resize == 1)
            # Find bounding box coordinates: ROW-->HIGHT
            min_row, min_col = np.min(ones_coordinates, axis=0)
            max_row, max_col = np.max(ones_coordinates, axis=0)
            # Get the center point and four sides
            mid_row = int((min_row + max_row) / 2); right_row = mask_resize.shape[0] - mid_row
            mid_col = int((min_col + max_col) / 2); right_col = mask_resize.shape[1] - mid_col
            # Get the min side length
            sides = [mid_row, right_row, mid_col, right_col]
            crop_size = np.min(sides)
        # Center Crop
        origin_crop = origin[mid_row-crop_size:mid_row+crop_size, mid_col-crop_size:mid_col+crop_size]
        assert origin_crop.shape[0]== origin_crop.shape[1], 'Not a square image!'
        print(f'{idx} Square Crop: {origin_crop.shape}')
        ########################Resize512/256########################
        # shrinking and save the image
        if resize512:
            origin_512 = cv2.resize(np.array(origin_crop), dsize=(512, 512), interpolation=cv2.INTER_AREA) # np.array, uint8
            cv2.imwrite(dst_path_512, origin_512)
        if resize256:
            origin_256 = cv2.resize(np.array(origin_crop), dsize=(256, 256), interpolation=cv2.INTER_AREA) # np.array, uint8
            cv2.imwrite(dst_path_256, origin_256)  

        # Read the binary lung mask
        # final_img = cv2.imread(dst_path_512, cv2.IMREAD_GRAYSCALE)
        # print(f'Final: {final_img.shape, type(final_img), final_img.dtype}') # HW, torch.uint8

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))

if __name__ == '__main__':
    main(args)