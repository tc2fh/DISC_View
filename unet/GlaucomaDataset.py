import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    def __init__(self, images_path, masks_path, img_filenames, mask_filenames, refuge=False, input_img_size=256):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames
        self.images = []
        self.masks = []

        # Pre-processing images: convert to RGB array, convert to tensor, resize
        for img in self.img_filenames:
            img_name = os.path.join(images_path, img)
            img = np.array(Image.open(img_name).convert('RGB'))
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.resize(
                img, size=(input_img_size, input_img_size), interpolation=Image.BILINEAR)
            self.images.append(img)

        # # Pre-processing masks: convert to array, create binary masks, convert to tensors, resize
        for mask in self.mask_filenames:
            mask_name = os.path.join(masks_path, mask)
            mask = np.array(Image.open(mask_name, mode='r'))

            # Create binary masks for optic disc and optic cup classes
            if refuge:
                od = (mask == 128).astype(np.float32)
                oc = (mask == 0).astype(np.float32)
            else:
                od = (mask > 0).astype(np.float32)
                oc = (mask > 1.).astype(np.float32)

            # Convert to tensor and add batch dimension
            od = torch.from_numpy(od[None, :, :])  # (1, Height, Width)
            oc = torch.from_numpy(oc[None, :, :])

            # Resize using nearest neighbor interpolation
            od = transforms.functional.resize(
                od, size=(input_img_size, input_img_size), interpolation=Image.NEAREST)
            oc = transforms.functional.resize(
                oc, size=(input_img_size, input_img_size), interpolation=Image.NEAREST)
            self.masks.append(torch.cat([od, oc], dim=0))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        return self.images[idx], self.img_filenames[idx], self.masks[idx], self.mask_filenames[idx]


class GlaucomaDatasetBoundingBoxes(GlaucomaDataset):
    def __init__(self, images_path, masks_path, img_filenames, mask_filenames, bbox_df, input_img_size):
        super().__init__(images_path, masks_path, img_filenames, mask_filenames, input_img_size)
        self.bbox_df = bbox_df  # Store bounding box dataframe

    def __getitem__(self, idx):
        image, img_filename, mask, mask_filename = super().__getitem__(idx)
        
        # Extract the bounding box coordinates for the current image.
        # This returns a list: [tensor_x1, tensor_y1, tensor_x2, tensor_y2]
        bbox_coords_list = self.bbox_df[self.bbox_df["image_path"] == img_filename][["x1", "y1", "x2", "y2"]].values[0]
        
        # Convert each coordinate to an integer tensor if needed
        bbox_list = [torch.tensor(coord, dtype=torch.int32) for coord in bbox_coords_list]
        
        # Stack the list into a single tensor of shape [num_boxes, 4]
        # First, reshape each tensor (if necessary) to ensure they have the same shape (e.g., [8])
        # Then stack along a new dimension and transpose so that each box is a row.
        bbox_tensor = torch.stack(bbox_list, dim=-1)  # This yields shape [8, 4]
        
        return {
            "image": image,
            "image_filename": img_filename,
            "mask": mask,
            "mask_filename": mask_filename,
            "bbox": bbox_tensor  # Now a tensor of shape [8, 4]
        }