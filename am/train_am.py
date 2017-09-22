import sys
import os
import argparse
import time
import random
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import cuda_functional as MF

from io_func.kaldi_feat import KaldiReadIn
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label, make_context, skip_frame
from io_func.kaldi_io_parallel import KaldiDataReadParallel

import pdb
#####
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.feadim
        self.n_cell=args.hidnum
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.n_V = args.statenum
        if args.lstm:
            self.rnn = nn.LSTM(self.n_d, self.n_cell,
                self.depth,
                dropout = args.rnn_dropout
            )
        else:
            self.rnn = MF.SRU(self.n_d, self.n_cell, self.depth,
                dropout = args.rnn_dropout,
                rnn_dropout = args.rnn_dropout,
                use_tanh = 0
            )
        self.output_layer = nn.Linear(self.n_cell, self.n_V)

        self.init_weights()
        if not args.lstm:
            self.rnn.set_bias(args.bias)

    def init_weights(self):
        val_range =0.05 
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, x, hidden,lens):
        x=pack_padded_sequence(x,lens,batch_first=True)
        rnnout, hidden = self.rnn(x, hidden)
        output,_ = pad_packed_sequence(rnnout,batch_first=True)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_cell).zero_())
        if self.args.lstm:
            return (zeros, zeros)
        else:
            return zeros

    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

def train_model(epoch, model, train_reader):
    model.train()
    args = model.args
    
    train_reader.initialize_read(True)
    batch_size = args.batch_size
    lr = args.lr

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(size_average=True,ignore_index=-1)
    hidden = model.init_hidden(batch_size)
    i=0
    running_acc=0
    total_frame=0
    while True:
        feat,label,length = train_reader.load_next_nstreams()
        if length is None or label.shape[0]<args.batch_size:
            break
        else:
            x, y =  Variable(torch.from_numpy(feat)).cuda(), Variable(torch.from_numpy(label).long()).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
                else Variable(hidden.data)

            model.zero_grad()
            output, hidden = model(x, hidden,length)
            assert x.size(0) == batch_size
            loss = criterion(output, y.view(-1))
            _,predict = torch.max(output,1)
            correct = (predict == y.view(-1)).sum()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            for p in model.parameters():
                if p.requires_grad:
                    if args.weight_decay > 0:
                        p.data.mul_(1.0-args.weight_decay)
                    p.data.add_(-lr, p.grad.data)

            if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
                sys.exit(0)
                return

            total_loss += loss.data[0] * sum(length)
            running_acc += correct.data[0]
            total_frame += sum(length)
            #print running_acc*1.0/total_frame
            i+=1
            if i%10 == 0:
                sys.stdout.write("time:{}, Epoch={},trbatch={},loss={:.4f},tracc={:.4f}\n".format(datetime.now(),epoch,i,total_loss/total_frame,\
                        running_acc*1.0/total_frame))
                sys.stdout.flush()

    return total_loss/total_frame,running_acc*1.0/total_frame

def eval_model(epoch,model, valid_reader):
    model.eval()
    args = model.args
    valid_reader.initialize_read(True)
    batch_size = args.batch_size
    total_loss = 0.0
    #unroll_size = model.args.unroll_size
    criterion = nn.CrossEntropyLoss(size_average=True,ignore_index=-1)
    hidden = model.init_hidden(batch_size)
    i=0
    total_frame=0
    cvacc=0
    while True:
        feat,label,length = valid_reader.load_next_nstreams()
        if length is None or label.shape[0]<args.batch_size:
            break
        else:
            x, y = Variable(torch.from_numpy(feat), volatile=True).cuda(), Variable(torch.from_numpy(label).long()).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
                else Variable(hidden.data)
            output, hidden = model(x, hidden,length)
            loss = criterion(output, y.view(-1))#/x.size(0)
            _,predict=torch.max(output,1)
            correct=(predict == y.view(-1)).sum()
            total_frame+= sum(length)
            total_loss += loss.data[0]*sum(length)
            cvacc +=correct.data[0]
            i+=1
            if i%10 == 0:
                sys.stdout.write("time:{}, Epoch={},cvbatch={},loss={:.4f},cvacc={:.4f}\n".format(datetime.now(),epoch,i,total_loss/total_frame,\
                        cvacc*1.0/total_frame))
                sys.stdout.flush()
    avg_loss = total_loss / total_frame
    return avg_loss,cvacc*1.0/total_frame

def main(args):
    train_read_opts={'label':args.trainlab,'lcxt':args.lcxt,'rcxt':args.rcxt,'num_streams':args.batch_size,'skip_frame':args.skipframe}
    dev_read_opts={'label':args.devlab,'lcxt':args.lcxt,'rcxt':args.rcxt,'num_streams':args.batch_size,'skip_frame':args.skipframe}

    kaldi_io_tr=KaldiDataReadParallel(args.train,train_read_opts)
    kaldi_io_dev=KaldiDataReadParallel(args.dev,dev_read_opts)
    model = Model(args)
    model.cuda()
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    model.print_pnorm()
    sys.stdout.write("\n")

    unchanged = 0
    best_dev = 1e+8
    #pdb.set_trace()
    for epoch in range(args.max_epoch):
        start_time = time.time()
        if args.lr_decay_epoch>0 and epoch>=args.lr_decay_epoch:
            args.lr *= args.lr_decay
        train_loss,tracc = train_model(epoch, model, kaldi_io_tr)
        dev_loss,devacc = eval_model(epoch,model, kaldi_io_dev)
        sys.stdout.write("Epoch={}  lr={:.4f}  train_loss={:.4f}  dev_loss={:.4f}  tracc={:.4f}  validacc={:.4f}"
                "\t[{:.4f}m]\n".format(
            epoch,
            args.lr,
            train_loss,
            dev_loss,
            tracc,
            devacc,
            (time.time()-start_time)/60.0
        ))
        model.print_pnorm()
        sys.stdout.flush()

        if dev_loss < best_dev:
            unchanged = 0
            best_dev = dev_loss
            start_time = time.time()
            sys.stdout.flush()
        else:
            unchanged += 1
        if unchanged >= 30: break
        sys.stdout.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--lstm", action="store_true")
    argparser.add_argument("--train", type=str, required=True, help="kaldi formate train scp file")
    argparser.add_argument("--dev", type=str, required=True, help="kaldi formate dev scp file")
    #argparser.add_argument("--test", type=str, required=True, help="test file")
    argparser.add_argument("--trainlab", type=str, required=True, help="kaldi formate train lab file")
    argparser.add_argument("--devlab", type=str, required=True, help="kaldi formate dev lab file")
    argparser.add_argument("--lcxt",type=int,default=0)
    argparser.add_argument("--rcxt",type=int,default=0)
    argparser.add_argument("--skipframe",type=int,default=0)
    argparser.add_argument("--statenum",type=int,default=4043)

    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--unroll_size", type=int, default=35)
    argparser.add_argument("--max_epoch", type=int, default=300)
    argparser.add_argument("--feadim", type=int, default=40)
    argparser.add_argument("--hidnum", type=int, default=512)
    argparser.add_argument("--dropout", type=float, default=0.5,
        help="dropout of word embeddings and softmax output"
    )
    argparser.add_argument("--rnn_dropout", type=float, default=0.2,
        help="dropout of RNN layers"
    )
    argparser.add_argument("--bias", type=float, default=-3,
        help="intial bias of highway gates",
    )
    argparser.add_argument("--depth", type=int, default=6)
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=175)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--clip_grad", type=float, default=5)

    args = argparser.parse_args()
    print (args)
    main(args)
