# Detecting Road Damage
## Setup
```
pip install -r requirements.txt
```
## About the Model
This Implementation is centered arount the YOLOv8 Model by ultralytics. The architecture is as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/222869864-1955f054-aa6d-4a80-aed3-92f30af28849.jpg"/>
YOLOv8-P5 model structure
</div>

|    |              from | n  |  params | module                                    |   arguments           | 
|--|-----------------|---|-----------|----------------------------------|-----------------------|     
 | 0  |                -1 | 1   |  2320 | ultralytics.nn.modules.Conv                 | [3, 80, 3, 2]         |        
 | 1  |                -1 | 1   | 115520 | ultralytics.nn.modules.Conv                |  [80, 160, 3, 2]       |        
 | 2  |                -1 | 3   | 436800  |ultralytics.nn.modules.C2f                 |  [160, 160, 3, True]    |       
 | 3  |                -1 | 1   | 461440  |ultralytics.nn.modules.Conv                |  [160, 320, 3, 2]        |      
 | 4  |                -1 | 6   |3281920  |ultralytics.nn.modules.C2f                 |  [320, 320, 6, True]      |     
 | 5  |                -1 | 1   |1844480  |ultralytics.nn.modules.Conv                |  [320, 640, 3, 2]         |     
 | 6  |                -1 | 6  |13117440  |ultralytics.nn.modules.C2f                 |  [640, 640, 6, True]      |     
 | 7  |                -1 | 1   |3687680  |ultralytics.nn.modules.Conv                |  [640, 640, 3, 2]         |     
 | 8  |                -1 | 3   |6969600  |ultralytics.nn.modules.C2f                 |  [640, 640, 3, True]      |     
 | 9  |                -1 | 1   |1025920  |ultralytics.nn.modules.SPPF                |  [640, 640, 5]            |     
 |10  |                -1 | 1   |      0  |torch.nn.modules.upsampling.Upsample       |  [None, 2, 'nearest']     |     
 |11  |           [-1, 6] | 1   |      0  |ultralytics.nn.modules.Concat              | [1]                       |    
 |12  |                -1 | 3   |7379200  |ultralytics.nn.modules.C2f                 |  [1280, 640, 3]           |     
 |13  |                -1 | 1   |      0  |torch.nn.modules.upsampling.Upsample       |  [None, 2, 'nearest']     |     
 |14  |           [-1, 4] | 1   |      0  |ultralytics.nn.modules.Concat              |  [1]                      |     
 |15  |                -1 | 3   |1948800  |ultralytics.nn.modules.C2f                 |  [960, 320, 3]            |     
 |16  |                -1 | 1   | 922240  |ultralytics.nn.modules.Conv                |  [320, 320, 3, 2]         |     
 |17  |          [-1, 12] | 1   |      0  |ultralytics.nn.modules.Concat              |  [1]                      |     
 |18  |              -1 | 3   |7174400  |ultralytics.nn.modules.C2f                  | [960, 640, 3]              |  
 |19  |               -1 | 1   |3687680 | ultralytics.nn.modules.Conv                |  [640, 640, 3, 2]            |  
 |20  |           [-1, 9] | 1   |      0 | ultralytics.nn.modules.Concat             |   [1]                       |    
 |21  |                 -1 | 3  | 7379200 | ultralytics.nn.modules.C2f               |    [1280, 640, 3]         |       
 |22  |        [15, 18, 21] | 1  | 8721820 | ultralytics.nn.modules.Detect           |     [4, [320, 640, 640]] |
## How to Train
### Pretraining
``
python yolo/pretrain.py
``
this uses the hyperparameters at `yolo/prehyperparams.yaml`
### Training on Norway Dataset
``
python yolo/train.py
``

this uses the hyperparameters at `yolo/hyperparams.yaml`
## How to Predict
The prediction API with image fragmentation and model ensembling can be imported from `yolo/predict`:

```python
from predict import get_prediction

model_files = [...]
models =  [YOLO(model_file, task="detect")  for model_file in model_files]
img = cv2.load('path/to/img.jpeg')
boxes =  get_prediction(
				models,
				img,
				use_fragmentation=True,
				fragment_size=640,
				intersection_th=0.3,
				)
```
 
