import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from LitModel import LitModel
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

img_size = (480,480)
bs = 128

imgnet_train = "/home/ubuntu/datasets/imagenet/train/"
imgnet_val = "/home/ubuntu/datasets/imagenet/val/"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = "best-checkpoint-full-imgnet-augment-new"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="checkpoints",
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = datasets.ImageFolder(imgnet_train, transform=train_transform)
ds_val = datasets.ImageFolder(imgnet_val, transform=val_transform)
train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=bs, num_workers=32)
val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=bs, num_workers=32)

torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(accelerator="gpu", devices=2, precision="16-mixed", max_epochs=10, strategy="ddp", callbacks=[checkpoint_callback])
model = LitModel(num_classes=1000)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# lightning deepspeed has saved a directory instead of a file
save_path = f"checkpoints/{checkpoint_name}.ckpt"
output_path = f"{checkpoint_name}.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)