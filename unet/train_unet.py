import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2

from unet import UNet


def dice_coefficient(targets, preds, smooth=1e-6):
    intersection = torch.sum(preds * targets, dim=(2, 3))
    dice = (2. * intersection + smooth) / (
        torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3)) + smooth)
    return dice.mean()


class GlaucomaDataset(Dataset):
    def __init__(self, images_path, masks_path, img_filenames, mask_filenames, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_file = os.path.join(self.images_path, self.img_filenames[idx])
        mask_file = os.path.join(self.masks_path, self.mask_filenames[idx])

        image = np.array(Image.open(img_file).convert('L'))
        mask = np.array(Image.open(mask_file, mode='r'))

        od = (mask > 0).astype(np.float32)
        oc = (mask > 1).astype(np.float32)
        mask = np.stack([od, oc], axis=-1)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask.transpose(2, 0, 1))

        return image.float(), mask.float()


class GlaucomaDataModule(pl.LightningDataModule):
    def __init__(self, images_path, masks_path, batch_size=8, num_workers=4, input_size=256):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

    def setup(self, stage=None):
        img_files = sorted(os.listdir(self.images_path))
        mask_files = sorted(os.listdir(self.masks_path))

        train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
            img_files, mask_files, test_size=0.3, random_state=42)
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(
            temp_imgs, temp_masks, test_size=0.5, random_state=42)

        train_tfms = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2()
        ])

        val_tfms = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2()
        ])

        self.train_ds = GlaucomaDataset(self.images_path, self.masks_path,
                                        train_imgs, train_masks, transforms=train_tfms)
        self.val_ds = GlaucomaDataset(self.images_path, self.masks_path,
                                      val_imgs, val_masks, transforms=val_tfms)
        self.test_ds = GlaucomaDataset(self.images_path, self.masks_path,
                                       test_imgs, test_masks, transforms=val_tfms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)


class UNetModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=1)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        dice = dice_coefficient(masks, preds)
        return loss, dice

    def training_step(self, batch, batch_idx):
        loss, dice = self.step(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_dice', dice, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice = self.step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_dice', dice, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, dice = self.step(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_dice', dice, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--images', required=True, help='Path to training images')
    parser.add_argument('--masks', required=True, help='Path to training masks')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--output-dir', default='training_logs')
    args = parser.parse_args()

    dm = GlaucomaDataModule(args.images, args.masks,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            input_size=args.input_size)
    model = UNetModule(lr=args.lr)

    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name='tensorboard')
    csv_logger = CSVLogger(save_dir=args.output_dir, name='csv')

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=[tb_logger, csv_logger],
                         accelerator='auto')
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    ckpt_path = os.path.join(args.output_dir, 'unet_model.ckpt')
    trainer.save_checkpoint(ckpt_path)


if __name__ == '__main__':
    main()
