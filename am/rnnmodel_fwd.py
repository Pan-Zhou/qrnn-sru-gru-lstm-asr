import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnmodel import Model
from io_func.kaldi_feat import KaldiReadIn KaldiWriteOut
from io_func.kaldi_io_parallel import KaldiDataReadParallel

def main(args):
    dev_read_opts={'lcxt':args.lcxt, 'rcxt':args.rcxt,
            'num_streams':1, 'skip_frame':0}

    kaldi_io_infeat  = KaldiDataReadParallel(args.in_feat,dev_read_opts)
    kaldi_io_outfeat = KaldiWriteOut(args.out_feat)

    model = Model(args)
    model.cuda()
    
    model.load_state_dict(torch.load(args.model)['state_dict'])
    model.eval()
    kaldi_io_infeat.initialize_read(True)
    batch_size = 1
    i = 0
    while True:
        utt_id,utt_mat = kaldi_io_infeat.read_next_utt()###utt_mat.shape: T x D
        if utt_mat is None:
            break
        else:
            utt_mat = np.expand_dims(utt_mat, axis=1)###utt_mat.shae:T x 1 x D
            x = Variable(torch.from_numpy(x),volatile=True).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data),Variable(hidden[1].data)) if args.rnn_type =='lstm' else Variable(hidden.data)

            output,_ = model(x, hidden)

            if args.apply_logsoftmax:
                output = nn.LogSoftmax()(output)
            kaldi_io_outfeat.write_kaldi_mat(uttid,output)
            i += 1
        if i%100 ==0
            print("processed {} utterance".format(i))

    kaldi_io_outfeat.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='rnnmodel fwd inference')
    argparser.add_argument("--model", default="", help="location of model to be loaded")
    argparser.add_argument("--in_feat", default="", help="kaldi feat input rspecifier")
    argparser.add_argument("--out_feat", default="", help="kaldi feat output wspecifier")
    argparser.add_argument("--feadim", type=int, help="input feat dimension")
    argparser.add_argument("--statenum", type=int, help="output feat dimension")
    argparser.add_argument("--depth", type=int, default=3,help="number of hidden layers")
    argparser.add_argument("--rnn_type", default = "lstm",help="rnntype, lstm or gru or sru")
    argparser.add_argument("--dropout", type=float, default= 0.0,help="dropout probability")
    argparser.add_argument("--hidnum", type=int, default=512, help="nodes of hidden layers")
    argparser.add_argument("--bias", type = float, default=0.0, help="bias value of sru highway")
    argparser.add_argument("--apply_logsoftmax", action="store_true",help="output logsoftmax of posterior")
    args = argparser.parse_args()
    print(args)
    main(args)
