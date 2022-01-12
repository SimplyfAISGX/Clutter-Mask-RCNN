#!/usr/bin/env python3
"""
@Filename:    clutter_mask_rcnn.py
@Author:      dulanj
@Time:        12/01/2022 17:50
"""
from datetime import datetime

import PIL
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as Tf
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ycb = [
    "003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf",
    "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"
]


class ClutterMaskRCNN():
    def __init__(self, model_path='clutter_maskrcnn_model.pt'):
        num_classes = len(ycb) + 1
        model = self.get_instance_segmentation_model(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(self.device)
        self.model = model

    def get_instance_segmentation_model(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model

    def predict(self, image: PIL.Image, threshold=0.75):
        tic = datetime.utcnow()
        # pick one image from the test set (choose between 9950 and 9999)
        img = image.convert("RGB")

        with torch.no_grad():
            prediction = self.model([Tf.to_tensor(img).to(self.device)])

        scores = prediction[0]['scores'].cpu().detach().numpy()
        selection = scores > threshold
        labels = prediction[0]['labels'].cpu().detach().numpy()[selection]
        boxes = prediction[0]['boxes'].cpu().detach().numpy()[selection]
        masks = prediction[0]['masks'].cpu().detach().numpy()[selection]

        results = []
        for score, label, box, mask in zip(selection, labels, boxes, masks):
            results.append({
                "confidence": str(score),
                "label": label,
                "points": box,
                "type": "rectangle",
            })
        toc = datetime.utcnow()
        prediction = {
            'results': results,
            'start': tic.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'end': toc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'pred_time_s': round((toc - tic).total_seconds(), 3)
        }
        return prediction


if __name__ == '__main__':
    model = ClutterMaskRCNN()
    image = Image.open('/home/dulanj/Pictures/test/children-playing-outdoors.jpg')
    prediction = model.predict(image)
    img1 = ImageDraw.Draw(image)
    for result in prediction["results"]:
        print(result)
        x1, y1, x2, y2 = result["points"]
        img1.rectangle((x1, y1, x2, y2), outline='red')
    plt.imshow(image)
    plt.show()
