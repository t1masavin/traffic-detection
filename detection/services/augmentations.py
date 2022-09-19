import albumentations as A

def get_training_augmentation():
    train_transform = [

        A.GaussNoise (p=0.2),
        A.PadIfNeeded(min_height=640, min_width=640, always_apply=True, border_mode=0),

        A.OneOf([
            A.RandomSizedBBoxSafeCrop(width=240, height=120),
            A.RandomSizedBBoxSafeCrop(width=320, height=240),
            A.RandomSizedBBoxSafeCrop(width=480, height=320),
            A.RandomSizedBBoxSafeCrop(width=640, height=480),
        ], p=1
        ),
        A.Resize(640, 480),
        A.HorizontalFlip(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.8,
        ),
        A.Normalize(),
    ]
    return A.Compose(train_transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.5))


def get_val_augmentation():
    val_transform = [
        A.Resize(width=640, height=480),
        A.Normalize(),
    ]
    return A.Compose(val_transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_infer_augmentation():
    infer_transform = [
        A.Resize(width=640, height=480),
        A.Normalize(),
    ]
    return A.Compose(infer_transform)