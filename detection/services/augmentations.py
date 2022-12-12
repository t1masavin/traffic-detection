import albumentations as A


def get_training_augmentation(width=640, height=720):
    train_transform = [

        # A.GaussNoise(p=0.2),
        A.PadIfNeeded(min_height=height, min_width=width,
                      always_apply=True, border_mode=0),

        A.OneOf([
            A.RandomSizedBBoxSafeCrop(width=320, height=320),
            A.RandomSizedBBoxSafeCrop(width=480, height=480),
        ], p=0.4
        ),
        A.Resize(width=width, height=height),
        A.HorizontalFlip(p=0.3),

    ]
    return A.Compose(
        train_transform, bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['category_ids'],))


def get_val_augmentation(width=640, height=720):
    val_transform = [
        A.PadIfNeeded(min_height=height, min_width=width,
                      always_apply=True, border_mode=0),
        A.Resize(width=width, height=height),
        # A.Normalize(),
    ]
    return A.Compose(val_transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def get_infer_augmentation(width=640, height=720):
    infer_transform = [
        A.Resize(width=width, height=height),
        # A.Normalize(),
    ]
    return A.Compose(infer_transform)
