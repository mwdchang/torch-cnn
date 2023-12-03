import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

################################################################################
# Load training directory
################################################################################
def create_dataset(directory):
    data_dir = directory
    dataset = ImageFolder(data_dir,transform = transforms.Compose([
        transforms.Resize((150, 150)),transforms.ToTensor()
    ]))

    # What classes are there
    print(f"Classes {dataset.classes}")

    # Print out data for the first image
    (img, label) = dataset[0]
    print(label, img.shape)

    return dataset


################################################################################
# Split into training and validation
################################################################################
def split_dataset(dataset):
    batch_size = 128
    val_size = 2
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dl = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    # train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
    # validation_dl = DataLoader(val_data, batch_size * 2, num_workers = 0, pin_memory = True)
    return (train_dl, validation_dl)


def load_images(directory):
    dataset = create_dataset(directory)
    train_dl, validation_dl = split_dataset(dataset)
    return train_dl, validation_dl
