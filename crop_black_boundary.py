'''
Pre-processing
Image:
Step1: Boundary crop and insert the new image size into .csv
'''
import pandas as pd
import cv2
import os
import time

root_dir = "/data/epione/user/huili/MIMIC-CXR-JPG/files/mimic-cxr-jpg/2.0.0/files"
new_dir = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/files"
metadata_df_path = "/data/epione/user/huili/MIMIC-CXR-JPG/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
metadata_df_cropped_path = "/data/epione/user/huili/MIMIC-CXR-JPG-cropped/mimic-cxr-2.0.0-metadata-cropped.csv"

start_time = time.time()

metadata_df = pd.read_csv(metadata_df_path)
df = metadata_df[["dicom_id", "subject_id", "study_id", 
            "PerformedProcedureStepDescription", "ViewPosition"]] # len=377110
num = len(df)

Rows_list = []
Columns_list = []

for idx in range(num):
  row = df.iloc[idx]
  src_path = f'{root_dir}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
  dst_dir =  f'{new_dir}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}'
  dst_path = f'{dst_dir}/{row["dicom_id"]}.jpg'
  # print(f'src_path: {src_path}')
  # print(f'dst_path: {dst_path}')

  image_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

  # Apply thresholding
  _, thresholded = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)
  # Find contours
  contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Find the largest contour
  largest_contour = max(contours, key=cv2.contourArea)
  # Get the bounding box of the largest contour
  x, y, w, h = cv2.boundingRect(largest_contour)
  # Crop the image
  cropped_image = image_gray[y:y+h, x:x+w]
  print(f'image_gray: {image_gray.shape}, cropped_image: {cropped_image.shape}')
  Rows_list.append(cropped_image.shape[0])
  Columns_list.append(cropped_image.shape[1])
  
  # Save the image
  if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
  cv2.imwrite(dst_path, cropped_image)

df.insert(loc=5, column="Rows", value=Rows_list)
df.insert(loc=6, column="Columns", value=Columns_list)
# print(df)
df.to_csv(metadata_df_cropped_path, index=False)
print('Time {:.3f} min'.format((time.time() - start_time) / 60)) # Time 2361.116 min