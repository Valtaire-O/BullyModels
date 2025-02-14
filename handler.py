from ultralytics import YOLO
from typing import Dict, List, Any
import ast
from  PIL import Image,ImageChops
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import io
import os
import torch
from torchvision import transforms
import numpy as np

class EndpointHandler():

    def __init__(self, path=""):
        # Preload various  models.

        self.logoV8 = YOLO(os.path.join(path, "best.pt"))
        self.detectorV8 = YOLO(os.path.join(path, "yolov8n.pt"))
        self.resnet101 =torch.load(os.path.join(path, 'resnet101.pth'))
        self.resnet101.eval()
        self.resnet101 = torch.nn.Sequential(*list(self.resnet101.children())[:-1])

        # Define transformations
        self.transform = transforms.Compose(
            [  # seems to work better at 224 vs 256
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],

                ),  # transforms.GaussianBlur(kernel_size=(7, 13))
            ]
        )

    def __call__(self, data):

        if 'input' not in data['inputs'].keys():
            return {'status': 'error', "message": "missing 'input' key"}
        if 'task' not in data['inputs'].keys():
            return {'status': 'error', "message": "missing 'task' key"}
        task = data['inputs']['task']

        if task == 'batch_classify':
            if not isinstance(data['inputs']['input'], list):
                return {'status': 'error', "message": "input for batch classification must be a list"}

            return self.batch_classify(data['inputs']['input'])

        # attempt to open image
        input = self.open_img(data['inputs']['input'])
        if isinstance(input, dict):
            return  input

        if task == 'classify':
            detect_faces = self.detect(input)
            if detect_faces['prediction'] =='person':
                return {"prediction": "not_logo", "class": 1, "status": "success"}
            inference = self.classify(input)
            return inference


        if task == 'extract':
            inference = self.extract_features(input)
            return inference
        else:
            return {'status': 'error', "message": "task argument must be 'classify' or 'extract"}

    def batch_classify(self,data):
        for d in data:
            '''call open image'''
            img_content = d['content']
            img = self.open_img(img_content)
            if isinstance(img, dict):
                d['response'] = img
                continue

            detect_faces = self.detect(img)
            if detect_faces['prediction'] == 'person':
                d['response'] ={"prediction": "not_logo", "class": 1, "status": "success"}
                continue
            inference = self.classify(img)
            d['response'] = inference
        return data



    def open_img(self,input):
        try:
            image_bytes = ast.literal_eval(input)
            return  PIL.Image.open(io.BytesIO(image_bytes))
        except:
            return {'status': 'error', "message": "failed to open image due to the following error"}

    def detect(self, input):
        cls_names = self.detectorV8.names
        results = self.detectorV8.predict(input)
        '''Loop through the results and get the labels of objects detected'''
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            unique_cls_set = list(set([int(c) for c in boxes.cls]))
            objects_detected = [cls_names[i] for i in unique_cls_set]
            if 'person' in objects_detected:
                return {'prediction': 'person', 'class': 0, 'status': 'success'}
            return {'prediction': 'none', 'class': 1, 'status': 'success'}

    def classify(self, input):
        '''inference from binary classifier'''
        cls_names = self.logoV8.names
        result = self.logoV8.predict(input)
        output = result[0].probs.top1
        prediction = cls_names[output]
        return {"prediction": prediction, "class": output, 'status': 'success'}

    def extract_features(self,input):
        img = input.convert('RGB')
        new_img_t = self.transform(img)
        new_batch_t = torch.unsqueeze(new_img_t, 0)
        with torch.no_grad():
            embedding = self.resnet101(new_batch_t).numpy().tolist()
        return {"embedding": embedding,  'status': 'success'}

import json
from time import perf_counter
with open('test_batch.json') as f:
    data = json.load(f)
start = perf_counter()
new_data = EndpointHandler().__call__(data)
stop = perf_counter()
print(f'finished in {stop-start}')
for d in new_data:
    print(d['response'])