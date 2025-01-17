from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
import torch
import cv2
import os


class InferImages:
    '''
    Image preprocessing class with FIFO

    ...

    Attributes
    ----------
    batch_size: int
        image batch size
    transforms: albumentations.Compose
        image transformations.

    Methods
    -------
    append:
        Adding an element to the beginning of a collection
    __call__:
        Pop batched elements 

    '''

    def __init__(self, batch_size, transforms=None):
        self.batch_size = batch_size
        self.transforms = transforms
        self.collection = list()

    def append(self, item):
        assert isinstance(item, list), 'TypeError! The element must be a list.'
        self.collection.extend(item)

    def __call__(self):
        len_collection = len(self.collection)
        assert len_collection >= 1, 'Empty Collection! You should add items before the calling.'
        if len_collection > self.batch_size:
            len_collection = self.batch_size
        list_images = self.collection[-len_collection:]
        del self.collection[-len_collection:]
        list_images = [self.__read_image(img_path) for img_path in list_images]
        return list_images

    def __read_image(self, img_path):
        img = cv2.imread(img_path)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = self.__to_torch(img)
        return img

    def __to_torch(self, img):
        return torch.as_tensor(img).permute(2, 0, 1).float() / 255.


class DetectionDataset(Dataset):
    def __init__(self, data, names, root, transforms=None):
        self.data = data
        self.root = root
        self.transforms = transforms
        self.names = names

    def __getitem__(self, idx):
        name = self.names[idx]
        df = self.data[self.data['image'] == name]
        img_path = os.path.join(self.root, name)
        img = cv2.imread(img_path) / 255.

        bboxes = []
        labels = []
        for index, row in df.iterrows():
            bboxes.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
            labels.append(row['class'])

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bboxes, category_ids=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['category_ids']
        target = {}
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        img = torch.as_tensor(img).permute(2, 0, 1) 
        return img.to(torch.float16), target

    def __len__(self):
        return len(self.names)


def collate_fn(batch):
    return tuple(zip(*batch))


class FasterRCNN(LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = 1e-3
        self.metric = MeanAveragePrecision()

    def forward(self, imgs, targets=None):
        self.detector.eval()
        return self.detector(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.3, patience=3), 'monitor': 'train_loss_step', }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        loss_dict = self.detector(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss_step', loss.detach(), on_step=True)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        img, boxes = batch
        pred_boxes = self.forward(img)
        self.val_loss = self.metric(pred_boxes, boxes)
        self.log('val_step', self.val_loss['map'], on_step=True)
        return {"val_loss": self.val_loss['map']}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val.detach()}
        # Логи валидационных эпох для tensorboard
        self.log('val_epoch_total_step', log_dict['val_loss'], on_epoch=True)
        return log_dict['val_loss']
