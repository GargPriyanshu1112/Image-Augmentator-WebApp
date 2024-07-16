import io
import cv2
import zipfile
import numpy as np
from PIL import Image
import albumentations as A


def get_image_transformer(
    checkbox_Rotation,
    checkbox_HFlip,
    checkbox_VFlip,
    checkbox_Zoom,
    checkbox_ShiftRGB,
    checkbox_ColorJitter,
):
    transformations = []

    if checkbox_Rotation:
        transformations.append(
            A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT)
        )
    if checkbox_HFlip:
        transformations.append(A.HorizontalFlip(p=0.5))
    if checkbox_VFlip:
        transformations.append(A.VerticalFlip(p=0.1))
    if checkbox_Zoom:
        transformations.append(A.RandomResizedCrop(height=224, width=24, p=1.0))
    if checkbox_ShiftRGB:
        transformations.append(A.RGBShift(p=0.5))
    if checkbox_ColorJitter:
        transformations.append(A.ColorJitter(p=0.5))

    transformer = A.Compose(transformations)
    return transformer


def get_image_buffer(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, "PNG")
    img_buffer.seek(0)
    return img_buffer


def get_zip_buffer(images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            name = f"img_{i+1}.png"
            data = get_image_buffer(img).read()
            zip_file.writestr(name, data)
    zip_buffer.seek(0)
    return zip_buffer


def get_augmentations(transformer, files, num_augs_per_image):
    augmentations = []
    for file in files:
        image = Image.open(file)
        image = np.array(image)
        for i in range(num_augs_per_image):
            augmented_image = transformer(image=image)["image"]
            augmented_image = Image.fromarray(augmented_image)
            augmentations.append(augmented_image)

    zip_buffer = get_zip_buffer(augmentations)
    return zip_buffer


def is_num_augs_valid(num_augs):
    if num_augs != "" and int(num_augs) > 0 and int(num_augs) <= 100:
        return 1
    else:
        return 0
