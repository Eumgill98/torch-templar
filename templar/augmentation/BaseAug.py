import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseAug:
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose(
            [
                A.Resize(resize, resize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]

