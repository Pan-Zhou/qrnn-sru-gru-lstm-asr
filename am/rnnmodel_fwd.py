import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnmodel import Model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='rnnmodel fwd inference')
    argparser.add_argument()
    args = argparser.parse_args()
    print(args)
    main(args)
