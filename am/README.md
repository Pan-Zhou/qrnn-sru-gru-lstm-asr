
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
  

  ```
  - To train 3-layer LSTM and 7-layer SRU acoustic models:
  ```
	run-am.sh
```
  
