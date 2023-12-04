import torch
from model import NaturalSceneClassification
# from loader import load_images
from torchvision import transforms
from PIL import Image

PATH = "saved-model.pt"


def predict_img_class(img, model):
    """ Predict the class of image and Return Predicted Class"""
    # img = to_device(img.unsqueeze(0), device)
    prediction = model(img)
    _, preds = torch.max(prediction, dim=1)
    return preds[0].item()


if __name__ == '__main__':
    model = NaturalSceneClassification()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    print("\n\nTesting predicting individual images")
    # img = Image.open("../../IMG_7707.jpg")
    img = Image.open("./images/good/IMG_3819.jpeg")
    img = transforms.Resize((150, 150))(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    print(f"Predicted Class : {predict_img_class(img, model)}")
