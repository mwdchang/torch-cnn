# See https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

import torch
import numpy as np
from model import NaturalSceneClassification
from loader import load_images
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, learning_rate, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), learning_rate)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def predict_img_class(img, model):
    """ Predict the class of image and Return Predicted Class"""
    # img = to_device(img.unsqueeze(0), device)
    prediction = model(img)
    _, preds = torch.max(prediction, dim=1)
    # return dataset.classes[preds[0].item()]
    return preds[0].item()


num_epochs = 2
optimizer_func = torch.optim.Adam
learning_rate = 0.001

if __name__ == '__main__':
    # initialize
    model = NaturalSceneClassification()
    (train_dl, validation_dl) = load_images("./images")
    history = fit(num_epochs, learning_rate, model, train_dl, validation_dl, optimizer_func)

    ################################################################################
    # TODO
    # - Save model parameters
    # - Load model parameters
    ################################################################################

    ################################################################################
    # Prediction
    ################################################################################
    print("\n\nTesting predicting individual images")
    img = Image.open("../../IMG_7707.jpg")
    img = transforms.Resize((150, 150))(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    print(f"Predicted Class : {predict_img_class(img, model)}")
