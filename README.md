# IMC
---
### INTRA-MODAL CONSTERAINT LOSS FOR IMAGE-TEXT RETRIEVAL
This is code for IMC work.
---
## Requirements and Installation

* Python3
* PyTorch
* NumPy 
* TensorBoard
* pycocotools
* torchvision
* torchtext
* matplotlib
* nltk:
---

## Download Data
Download the dataset files (MS-COCO and Flickr30K) in /data.
---
## Training new models
```bash
python train.py --data_path "$DATA_PATH" --data_name coco --logger_name 
runs/coco_imc --max_violation
```
---

## Evaluate pre-trained models
```python
python -c "\
from vocab import Vocabulary
import evaluation
evaluation.evalrank('$RUN_PATH/coco_imc/model_best.pth.tar', data_path='$DATA_PATH', split='test')"
```

## Acknowledgements
Our code is besed on VSE++. We thank to the authors for releasing codes.
