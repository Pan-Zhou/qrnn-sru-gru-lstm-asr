
Code used for neural acoustic modeling. 
In the experiments, we used identity activation `--use_tanh 0` and set highway gate bias to `-3`.
These choices are found to produce better results.

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
		--train train.scp    # run with default options, 6 SRU layers  
		--dev valid.scp
		--trainlab tr.labels
		--devlab cv.labels

  ```
  
