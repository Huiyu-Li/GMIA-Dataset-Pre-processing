'''
Pre-processing with pretrained Unet
And disaply the results for each step
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
'''

import torch
import torchvision
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont 
import torchvision.transforms as transforms
from src.models import UNet, PretrainedUNet
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib
from scipy import ndimage

def crop_black_boundary(origin):
  # Apply thresholding
  _, thresholded = cv2.threshold(origin, 10, 255, cv2.THRESH_BINARY)
  # Find contours
  contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Find the largest contour
  largest_contour = max(contours, key=cv2.contourArea)
  # Get the bounding box of the largest contour
  x, y, w, h = cv2.boundingRect(largest_contour)
  # Crop the image
  cropped_image = origin[y:y+h, x:x+w]
  return cropped_image

def display_single(image, title, name=None, cmap=None, ticks=0, fontsize=20, save_dir=None, study_id=None):
    plt.figure(figsize=(20, 10))
    plt.title(f"{title}: {image.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(image, cmap=cmap) # torch.Tensor, np.array
    if ticks:
        plt.colorbar(ticks=ticks)
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

def display_double(image1, image2, title1, title2, name=None, cmap='viridis', fontsize=20, save_dir=None, study_id=None):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"{title1}: {image1.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(np.array(image1), cmap=cmap)
    plt.subplot(1, 2, 2)
    plt.title(f"{title2}: {image2.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(np.array(image2), cmap=cmap)
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

def display_triple(image1, image2, image3, title1=None, title2=None, title3=None, 
            name=None, cmaps=['viridis']*3, ticks=0, fontsize=20, save_dir=None, study_id=None):
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 3, 1)
    plt.title(f"{title1}: {image1.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(np.array(image1), cmap=cmaps[0])
    if ticks:
        # set the position of colorbar: [left, bottom, width, hight]
        cax = fig.add_axes([ax1.get_position().x1, ax1.get_position().y0, 
                            0.01, ax1.get_position().height])
        plt.colorbar(ticks=ticks, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    plt.subplot(1, 3, 2)
    plt.title(f"{title2}: {image2.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(np.array(image2), cmap=cmaps[1])
    plt.subplot(1, 3, 3)
    plt.title(f"{title3}: {image3.shape}", fontdict={'fontsize': fontsize})
    plt.imshow(np.array(image3), cmap=cmaps[2])
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

def display_blend(image, mask, title=None, name=None, fontsize=30, save_dir=None, study_id=None):
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)
    if type(mask) == np.ndarray:
        mask = torch.from_numpy(mask)

    shape = image.shape
    mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros((1, shape[0], shape[1])),
            torch.stack([mask]),
            torch.zeros((1, shape[0], shape[1]))
        ]))
    pil_image = torchvision.transforms.functional.to_pil_image(torch.stack([image])).convert("RGB")
    plt.figure(figsize=(20, 10))
    plt.title(f"{title}: {shape}", fontdict={'fontsize': fontsize})
    img = Image.blend(pil_image, mask1, 0.2)
    plt.imshow(np.array(img))
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

def display_triple_blend(image, mask1, mask2, mask3, title1=None, title2=None, title3=None, name=None, fontsize=20, save_dir=None, study_id=None):
    image = image.cpu().detach()# HW, torch.float32
    shape = image.shape # HW
    mask1 = torch.argmax(mask1, dim=1).float() # 1HW, torch.int64->float32
    mask2 = torch.argmax(mask2, dim=1).float() # 1HW, torch.int64->float32
    # print(torch.unique(mask1), torch.unique(mask1))
    mask1 = mask1.cpu().detach() # 1HW, 
    mask2 = mask2.cpu().detach() # 1HW, 
    mask3 = mask3.cpu().detach() # 1HW, 
    mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros((1, shape[0], shape[1])),
            mask1,
            torch.zeros((1, shape[0], shape[1]))
        ]))
    mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros((1, shape[0], shape[1])),
            mask2,
            torch.zeros((1, shape[0], shape[1]))
        ]))
    mask3 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros((1, shape[0], shape[1])),
            mask3,
            torch.zeros((1, shape[0], shape[1]))
        ]))
    pil_image = torchvision.transforms.functional.to_pil_image(image + 0.5).convert("RGB")
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.title(f"{title1}", fontdict={'fontsize': fontsize})
    img = Image.blend(pil_image, mask1, 0.2)
    plt.imshow(np.array(img))
    plt.subplot(1, 3, 2)
    plt.title(f"{title2}", fontdict={'fontsize': fontsize})
    img = Image.blend(pil_image, mask2, 0.2)
    plt.imshow(np.array(img))
    plt.subplot(1, 3, 3)
    plt.title(f"{title3}", fontdict={'fontsize': fontsize})
    img = Image.blend(pil_image, mask3, 0.2)
    plt.imshow(np.array(img))
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

def display_bbox(image, mask, bbox, center, sides, title, name=None, fontsize=20, outline="red", width=4, save_dir=None, study_id=None):
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)
    if type(mask) == np.ndarray:
        mask = torch.from_numpy(mask)
    shape = image.shape
    pil_image = torchvision.transforms.functional.to_pil_image(torch.stack([image])).convert("RGB")
    # Create a PIL ImageDraw object
    draw = ImageDraw.Draw(pil_image) 
    # Draw the bounding box on the image, bbox[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
    draw.rectangle(bbox, outline=outline, width=width)
    # Draw the center point
    new_center = [center[0]-25, center[1]-25, center[0]+25, center[1]+25]
    draw.ellipse(new_center, fill=(0, 255, 255))
    # Draw the test points 
    draw.ellipse([0, center[1], 50, center[1]+50], fill=(255, 0, 0))
    draw.ellipse([shape[1]-50, center[1], shape[1], center[1]+50], fill=(0, 255, 0))
    draw.ellipse([center[0], 0, center[0]+50, 50], fill=(0, 0, 255))
    draw.ellipse([center[0], shape[0]-50, center[0]+50, shape[0]], fill=(255, 0, 255))
    # Draw the lines 
    draw.line((0, center[1],  center[0], center[1]), fill=(255, 0, 0), width = width) # left row
    draw.line((center[0], center[1], shape[1], center[1]), fill=(0, 255, 0), width = width) # right row
    draw.line((center[0], 0, center[0], center[1]), fill=(0, 0, 255), width = width) # left col
    draw.line((center[0], center[1], center[0], shape[0]), fill=(255, 0, 255), width = width)# right col

    # draw text, half opacity
    fnt = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf", size=200)
    draw.text((center[0]/2, center[1]), str(sides[2]), font=fnt, fill=(255, 0, 0))
    draw.text((shape[1]-center[0]/2, center[1]), str(sides[3]), font=fnt, fill=(0, 255, 0))
    draw.text((center[0], center[1]/2), str(sides[0]), font=fnt, fill=(0, 0, 255))
    draw.text((center[0], shape[0]-center[1]/2), str(sides[1]), font=fnt, fill=(255, 0, 255))

    mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros((1, shape[0], shape[1])),
            torch.stack([mask]),
            torch.zeros((1, shape[0], shape[1]))
        ]))
    
    plt.figure(figsize=(20, 10))
    plt.title(f"{title}: {shape}", fontdict={'fontsize': fontsize})
    img = Image.blend(pil_image, mask1, 0.2)
    plt.imshow(np.array(img))
    plt.savefig(f'{save_dir}/{study_id}/{study_id}-{name}.jpg', bbox_inches='tight')
    plt.close()

class MIMIC(Dataset):
    def __init__(self, 
                 root_dir,
                 csv_path, 
                 selected_rows = [0],# [0], [1, 3, 5, 7]
                 use_frontal=True,
                 boundary_crop=False,
                 save_dir = None,
                 ):
        
        self.root_dir = root_dir
        self.boundary_crop = boundary_crop
        self.save_dir = save_dir

        # load data from csv
        self.df = pd.read_csv(csv_path)# len=377110
        if use_frontal:
            # filtering by ViewPosition
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])] # len=243334
        if selected_rows:
            self.df = self.df.iloc[selected_rows] 
        self._num_images = len(self.df)
        

    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):
          row = self.df.iloc[idx]
          image_path = f'{self.root_dir}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
          print('')
          print(f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg')
          
          study_id = f's{str(row["study_id"])}'
          parent_dir = f'{self.save_dir}/{study_id}'
          if not os.path.exists(parent_dir):
              os.makedirs(parent_dir)

          # Read the original chest X-ray image
          origin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # np.array, dtype=uint8
          
          if self.boundary_crop:
            print(f'Origin: {origin.shape, type(origin), origin.dtype}') # HW, torch.uint8
            display_single(origin.copy(), title='Origin', name='origin', cmap='gray', fontsize=30, save_dir=self.save_dir, study_id=study_id)
            ########################Boundary Crop########################
            origin = crop_black_boundary(origin) # np.array, uint8
            print(f'Boundary Crop: {origin.shape, type(origin), origin.dtype}')
            display_single(origin.copy(), title='Boundary Crop', name='boundary', cmap='gray', fontsize=30, save_dir=self.save_dir, study_id=study_id)

          # shrinking input image, [cv2.INTER_AREA for shrinking and cv2.INTER_LINEAR for zooming]
          input0 = cv2.resize(origin, dsize=(512, 512), interpolation=cv2.INTER_AREA) # np.array, uint8
          # ToTensor().normalize to [0., 1.]->[-1, 1]
          input = transforms.ToTensor()(input0) - 0.5 # PyTorch tensor [1, 512, 512], torch.float32

          return origin, input, study_id
    
start_time = time.time()
root_cropped_dir = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/files"
root_origin_dir = "/data/epione/user/huili/MIMIC-CXR-JPG/files/mimic-cxr-jpg/2.0.0/files"
df_copped_path = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/mimic-cxr-2.0.0-metadata-cropped.csv"
save_dir = "./images"

FP_filter = True; use_2v = False; boundary_crop = False
selected_rows = [162222+38, 162222+39]  # Change 10 to the desired length of your list

# [1, 3, 5, 7, 9,
# 11, 13, 15, 17, 19,
# 21, 23, 25, 27, 29,]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet_6v = PretrainedUNet(in_channels=1, out_channels=2,  batch_norm=True, upscale_mode="bilinear")
unet_6v.load_state_dict(torch.load(f'.src/trained_models/unet-6v.pt', map_location=torch.device("cpu")))
unet_6v.to(device)
unet_6v.eval()
if use_2v:
    unet_2v = UNet(in_channels=1, out_channels=2,  batch_norm=False, upscale_mode="nearest")
    unet_2v.load_state_dict(torch.load(f'.src/trained_models/unet-2v.pt', map_location=torch.device("cpu")))
    unet_2v.to(device)
    unet_2v.eval()

if boundary_crop: 
    root_dir = root_origin_dir
    selected_rows = [0]
else: 
    root_dir = root_cropped_dir
    selected_rows = selected_rows
dataloader =  DataLoader(MIMIC(root_dir = root_dir,
                        csv_path = df_copped_path, 
                        selected_rows = selected_rows,
                        use_frontal=True,
                        boundary_crop=boundary_crop,
                        save_dir = save_dir), 
            batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=True)

for idx, (origin, input, study_id) in enumerate(dataloader):
    study_id = study_id[0]
    origin = origin[0] # CHW->HW, torch.uint8
    if not boundary_crop:
        print(f'Origin: {origin.shape, type(origin), origin.dtype}') # HW, torch.uint8
        # display_single(origin.clone(), title='Origin', name='origin', cmap='gray', fontsize=30, save_dir=save_dir, tudy_id=study_id)
    print(f'Nework Input: {input.shape}, [{input.min()}, {input.max()}]')# NCHW, torch.float32
    # display_single(input[0][0].clone(), title='Nework Input', name='input', cmap='gray', fontsize=30, save_dir=save_dir, study_id=study_id)

    input = input.to(device)
    with torch.no_grad():
        # Mask from unet-6v
        mask_6v = unet_6v(input) # 12HW, torch.float32
        softmax_6v = torch.nn.functional.log_softmax(mask_6v, dim=1) # 12HW, torch.float32
        if use_2v: 
            # Mask from unet-2v
            mask_2v = unet_2v(input) # 12HW, torch.float32
            softmax_2v = torch.nn.functional.log_softmax(mask_2v, dim=1) # 12HW, torch.float32
            # Taking the element-wise maximum from the softmax probability maps
            composite_mask = torch.max(softmax_6v, softmax_2v)
            # Applying argmax to get the final segmentation mask
            mask = torch.argmax(composite_mask, dim=1).float() # 1HW, torch.int64->float32
            display_triple_blend(input[0][0].clone(), softmax_6v.clone(), softmax_2v.clone(), mask.clone(),
                            title1='Mask 6v', title2='Mask 2v', title3='Mask Final', 
                            name='masks', fontsize=30, save_dir=save_dir, study_id=study_id)
        else:
            mask = torch.argmax(softmax_6v, dim=1).float() # 1HW, torch.int64->float32

    mask = mask[0].cpu().detach().numpy()# HW, np.array, float32
    (row, col) = origin.shape
    if np.all(mask == 0): # Perform center crop with shorter side
        # Get the center point
        mid_row = int(row / 2)
        mid_col = int(col / 2)
        # Get the min side length
        crop_size = int(np.min([row, col])/2)
        print(f'Zero mask!, Crop Size: {crop_size*2}')
        mask_resize = np.zeros((row, col))
    else:    
        ########################Filter FPs in Mask########################
        if FP_filter:
            # Use connected component labeling from scipy.ndimage
            labeled_mask, num_labels = ndimage.label(mask)
            print(f'Labels: {num_labels}')
            if num_labels>2:
                # Calculate the size of each connected component
                sizes = ndimage.sum(mask, labeled_mask, range(num_labels + 1))
                # Find the indices of the two largest components
                largest_indices = np.argsort(sizes)[-2:]
                # Create an empty mask for the two largest components
                mask_array = np.zeros_like(mask)
                # Keep the two largest components
                for index in largest_indices:
                    mask_array[labeled_mask == index] = 1 # np.array [0,1], uint8
                
                cmap = matplotlib.cm.get_cmap('Paired_r', num_labels+1)
                ticks = range(num_labels+1)
                display_triple(labeled_mask, mask, mask_array, title1='Mask Labeled', title2='Mask', title3='Mask Final', 
                            name='masks', cmaps=[cmap, 'viridis', 'viridis'], ticks=ticks, fontsize=30, save_dir=save_dir, study_id=study_id)
            else: 
                mask_array = mask.copy()
        else: 
            mask_array = mask.copy()
        # Save the binary lung mask
        # cv2.imwrite(f'./images/{study_id}/{study_id}-mask_array.jpg', mask_array.astype(np.uint8))

        ########################Zooming Mask########################
        mask_resize = cv2.resize(mask_array, dsize=(col, row), interpolation=cv2.INTER_NEAREST) # np.array([0., 1.], float32)
        # Check size
        # print(f'Check Size: {origin.shape}, {mask_resize.shape}')
        assert origin.shape[0]== mask_resize.shape[0] and origin.shape[1]== mask_resize.shape[1], 'Orign and Mask size mismatch!'
        display_double(mask_array.copy(), mask_resize.copy(), title1='Mask', title2='Mask Resized', name='mask', 
                    cmap='viridis', fontsize=30, save_dir=save_dir, study_id=study_id)

        ########################Square Crop########################
        # find_bounding_box
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
        print(f'Crop Size: {crop_size*2}')
        # Get the squate crop size as the max size of bbox and min_sizd, in case the min_size<max_bos
        # crop_size = np.max([int((max_row-min_row)/2), int((max_col-min_col)/2), min_side])

        bbox = [min_col, min_row, max_col, max_row] # [x0, y0, x1, y1]
        center = [mid_col, mid_row]
        display_bbox(origin.clone(), mask_resize.copy(), bbox, center, sides, title='Square', name='bbox', 
                fontsize=30, outline=(0, 255, 255), width=10, save_dir=save_dir, study_id=study_id)
    
    # Center Crop
    origin_crop = origin[mid_row-crop_size:mid_row+crop_size, mid_col-crop_size:mid_col+crop_size]
    mask_crop = mask_resize[mid_row-crop_size:mid_row+crop_size, mid_col-crop_size:mid_col+crop_size]
    assert origin_crop.shape[0]== origin_crop.shape[1], 'Not a square image!'
    print(f'Square Crop: {origin_crop.shape, mask_crop.shape}')
    display_blend(origin_crop.clone(), mask_crop.copy(), title='SquareCrop Origin + Mask', name='square', 
                fontsize=25, save_dir=save_dir, study_id=study_id)
    ########################Resize512########################
    # shrinking the image
    origin_512 = cv2.resize(np.array(origin_crop), dsize=(512, 512), interpolation=cv2.INTER_AREA) # np.array, uint8
    # Cannot use advanced interpolation approch !!!
    mask_512 = cv2.resize(mask_crop, dsize=(512, 512), interpolation=cv2.INTER_NEAREST) # np.array([0., 1.], float32)
    print(f'Resize 512: {origin_512.shape, mask_512.shape}')
    display_single(origin_512.copy(), title='Resize512 Origin', name='resize512', cmap='gray', fontsize=25, save_dir=save_dir, study_id=study_id)
    
print('Time {:.3f} min'.format((time.time() - start_time) / 60))

# Save the binary lung mask
# cv2.imwrite(f'./images/{study_id}-mask.jpg', out)
# Read the binary lung mask
# lung_mask = cv2.imread('lung_mask.jpg', cv2.IMREAD_GRAYSCALE)

def crop_lung_region_cv2(original_image, lung_mask):
    # Find contours in the lung mask
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the bounding box of the largest contour (assuming it corresponds to the lung)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # return [min_row, min_col, max_row, max_col]
        print(f'[{x}, {y}, {x+w}, {y+h}]')
        # Crop the lung region from the original image
        lung_region = original_image[y:y+h, x:x+w]
        return lung_region
    else:
        return None
    
def crop_lung_region_PIL(original_image, lung_mask):
    # find_bounding_box
    mask_array = np.array(lung_mask) # tensor-->numpy
    # Find coordinates of ones pixels (lung region)
    ones_coordinates = np.argwhere(mask_array == 1)
    # Find bounding box coordinates: ROW-->HIGHT-->y
    min_row, min_col = np.min(ones_coordinates, axis=0)
    max_row, max_col = np.max(ones_coordinates, axis=0)

    # return [min_row, min_col, max_row, max_col]
    print(f'[{min_col}, {min_row}, {max_col}, {max_row}]')
    # Create a PIL ImageDraw object
    draw = ImageDraw.Draw(original_image) # [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
    # Draw the bounding box on the image
    draw.rectangle([min_col, min_row, max_col, max_row], outline="red", width=2)
    return original_image
