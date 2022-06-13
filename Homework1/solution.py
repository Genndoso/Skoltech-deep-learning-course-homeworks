# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys
sys.path.append('https://drive.google.com/drive/folders/16B_nAvO1jKg0YDf2rXIpzAvlC3fKMJwC?usp=sharing')
import torchvision
import torch
from torchvision import transforms,datasets
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import make_grid
from matplotlib.pyplot import imshow
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.

    """
    candidates = [transforms.GaussianBlur(3),
                  transforms.RandomAdjustSharpness(2),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(0),
                  transforms.RandomVerticalFlip()]

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(7),
            transforms.RandomHorizontalFlip(10),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            #   AddGaussianNoise(),
            # transforms.RandomErasing(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.229, 0.224, 0.225])
            # transforms.ToTensor(),
        ])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.229, 0.224, 0.225])
         ])

    batch_size = 200
    if kind == "train":
        data = datasets.ImageFolder(f'{path}{kind}', transform=train_transform)
        dataset = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, num_workers=2, shuffle=True)
    else:
        data = datasets.ImageFolder(f'{path}{kind}', transform=test_transform)
        dataset = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, num_workers=2, shuffle=True)

    return dataset


def get_model():
    device = torch.device("cuda" if use_cuda else "cpu")
    # model = torchvision.models.resnet34(pretrained=False)
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 200, bias=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(3, 3), bias=False)

    return model.to(device)


def get_modelCNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            # second layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # third layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            # fourth layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # fifth layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=20, stride=2),

            # sixth
            nn.Flatten(),
            nn.Linear(61952, 1024),
            nn.Dropout(0.3),

            # seventh
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),

            # eigths
            nn.Linear(1024, 200),

            nn.Softmax(dim=1)
        )

    return model.to(device)
    
    
def get_optimizer(model):
    lr = 0.008
    betas = (0.9, 0.999)
    eps = 1e-08
    # config = {'learning_rate' : 0.01, 'beta1': 0.89, 'beta2': 0.999, 'epsilon': 1e-8}
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=betas,eps=eps)
    return optimizer



def predict(model, batch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    y_pred = model(batch)
    return y_pred.softmax(1)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here

    file = "/content/drive/MyDrive/DL HW1/model.pth"
    num_epochs = 30
    writer = SummaryWriter()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    val_accuracy = 2.3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()


    for epoch in tqdm(range(num_epochs)):

      model.train(True)
      pbar = tqdm(train_dataloader)
      correct = 0
      processed = 0
      avg_loss = 0
      for i, (X_batch, y_batch) in enumerate(train_dataloader):
        # get samples
        X_batch_gpu, y_true_on_device = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        # Forward pass
        logits = model(X_batch_gpu)
        # Calculate loss
        loss = criterion(logits, y_true_on_device)
        loss.backward()
        optimizer.step()
        # if scheduler:
        #   scheduler.step()



        # Update pbar-tqdm
        pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # pred = y_pred_train.softmax(1).max(1)[1]
        correct += pred.eq(y_true_on_device.view_as(pred)).sum().item()
        # correct += (pred == target).sum().item()
        processed += len(X_batch_gpu)
        avg_loss += loss.item()



        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f} , Accuracy={100*correct/processed:0.2f}%')

        #    pbar_str = f'Loss={loss.item():0.5f} Batch_id={i} Accuracy={100*correct/processed:0.2f}%'
         #   pbar.set_description(desc= pbar_str)

            # save model
            save_checkpoint(model, optimizer, filename=file)


      avg_loss /= len(train_dataloader)
      avg_acc = 100*correct/processed

      print(f'Accuracy of the network on test images: {avg_acc} %')

      #Tensorboard
      writer.add_scalar('train loss', avg_loss, epoch)
      writer.add_scalar('train accuracy', avg_acc, epoch)


      #Validation
      model.train(False)
      val_acc, val_loss = validate(val_dataloader, model)
      writer.add_scalar('validation loss', val_loss, epoch)
      writer.add_scalar('validation accuracy', val_acc*100, epoch)
      scheduler.step()

      #early stopping
      if val_acc > val_accuracy:
        val_accuracy = val_acc
       # torch.save(model.state_dict(), "/content/drive/MyDrive/DL HW1/checkpoint.pth")
        torch.save(model.state_dict(), "checkpoint.pth")


def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    processed = 0
    avg_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_val = model(X_batch)
            loss = criterion(y_pred_val, y_batch)

            pred = y_pred_val.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # pred = y_pred_train.softmax(1).max(1)[1]
            correct += pred.eq(y_batch.view_as(pred)).sum().item()
            # correct += (pred == target).sum().item()
            processed += len(X_batch)
            avg_loss += loss.item()

        avg_loss /= len(dataloader)
        avg_acc = correct / processed
    return avg_acc, avg_loss


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here;
    md5_checksum = "d11f0fc3e27cc937c0e0b804089b5143"

    # Your code here;
  #  google_drive_link = "https://drive.google.com/file/d/1ByeD16e38edHixQlnPKK1rJhgp7bs8zN/view?usp=sharing"
    google_drive_link =  "https://drive.google.com/file/d/1KwwLk7pq4BuM_JtG4bmri2iGljwtPYQk/view?usp=sharing"
    return md5_checksum, google_drive_link

