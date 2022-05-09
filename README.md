# INTRA-MODAL CONSTERAINT LOSS FOR IMAGE-TEXT RETRIEVAL
---
### This is code for IMC work.
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
* nltk
---

## Download Data

Download the dataset files (MS-COCO and Flickr30K) in /data.

---
## Training new models
```bash
python train.py --data_path "$DATA_PATH" --data_name coco --logger_name runs/coco_imc --max_violation
 --num_epochs 30 --rnn_type LSTM --wordemb glove --use_bidirectional --cnn_type resnet152 --use_restval --il_measure l1 
```
---

## Evaluate pre-trained models
You can download our pre-trained models [coco_imc](https://drive.google.com/drive/folders/19m3E0DDYuEXV_C1quRGw0J2Kf_uqpdn2?usp=sharing) and [f30k_imc](https://drive.google.com/drive/folders/1vLWQNV1pzkHa06CfQbbrLp5SjjKkLlve?usp=sharing) in `RUN_PATH`.

```python
python -c "\
from vocab import Vocabulary
import evaluation
evaluation.evalrank('$RUN_PATH/f30k_imc/model_best.pth.tar', data_path='$DATA_PATH', split='test')"
```
To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`$RUN_PATH/coco_imc/model_best.pth.tar`.

## Acknowledgements
Our code is besed on VSE++. We thank to the authors for releasing codes.
