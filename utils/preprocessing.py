from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ]
)


def preprocess_image(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor


def tensor_to_numpy(tensor):

    return tensor.squeeze().permute(1, 2, 0).cpu().numpy()
