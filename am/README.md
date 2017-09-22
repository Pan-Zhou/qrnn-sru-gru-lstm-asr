
Code used for neural acoustic modeling. 

## How to run
  - Prepare feathure and label file in kaldi format

  - Make sure CUDA library path and `cuda_functional.py` is available to python. For example,
  ```python
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export PYTHONPATH=../
  ```
  
  - Run **train_am.py** and get the results.
  ```
    python train_am.py --help               # see all running options
  
    python train_am.py 
		--train data/cv.scp     
		--dev data/cv.scp
		--trainlab data/sort_cv.labels
		--devlab data/sort_cv.labels
		--lr 0.05
		--max_epoch 10
		--lr_decay_epoch 4
		--batch_size 1

  ```
  
