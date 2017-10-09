import sys
import os
import argparse
import time
import random
import math
from datetime import datetime
import numpy as np
from more_itertools import sort_together

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import cuda_functional as MF

from io_func.kaldi_feat import KaldiReadIn
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label, make_context, skip_frame
from io_func.kaldi_io_parallel import KaldiDataReadParallel

def CELOSS(output,label):
    mask = (label>=0)
    output =F.log_softmax(output)
    labselect = label + (label<0).long()
    select = -torch.gather(output,1,labselect.view(-1,1))
    losses = mask.float().cuda().view(-1,1)*select
    loss = torch.sum(losses)/torch.sum(mask.float())
    return loss

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.from_numpy(np.arange(0, max_len)).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length)
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def padding(data, max_seq_len, value, data_shape, dtype=np.float32):
    seq_len = data.shape[0]
    shape = data.shape
    if seq_len <= max_seq_len:
        data_pad = np.zeros(shape, dtype=dtype)
        interpNum = max_seq_len - seq_len
        data_pad[0:seq_len] = data
        zero_mat = np.full(data_shape, value)
        for i in range(seq_len, max_seq_len):
            data_pad[i] = zero_mat
    else:
        data_pad = data[0:max_seq_len]
    return data_pad

def pad_seq(seq, max_length, value):
    pad_length = max_length - len(seq)
    if pad_length > 0:
        pad = np.array([value for i in range(pad_length)], dtype=np.int32)
        for ele in pad:
            seq.append(ele)
    return seq

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
                dropout = args.rnn_dropout,batch_first=False
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
        #x=pack_padded_sequence(x, lens, batch_first=True)
        rnnout, hidden = self.rnn(x, hidden)
        #output,_ = pad_packed_sequence(rnnout, batch_first=True) 
        output = self.drop(rnnout)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_cell).zero_())
        if self.args.lstm:
            zeros1 = Variable(weight.new(self.depth,batch_size,self.n_cell).zero_())
            return (zeros, zeros1)
        else:
            return zeros

    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))


def train_model(epoch, model, train_reader, optimizer):
    model.train()
    args = model.args
    
    train_reader.initialize_read(True)
    batch_size = args.batch_size
    lr = args.lr

    total_loss = 0.0
    #criterion = nn.CrossEntropyLoss(size_average=False,ignore_index=-1)
    hidden = model.init_hidden(batch_size)
    i=0
    running_acc=0
    total_frame=0
    while True:
        feat,label,length = train_reader.load_next_nstreams()
        if length is None or label.shape[0]<args.batch_size:
            break
        else: 
            xt=np.copy(np.transpose(feat,(1,0,2)))
            yt=np.copy(np.transpose(label,(1,0)))
            x,y = torch.from_numpy(xt),torch.from_numpy(yt).long()
            x, y = Variable(x.contiguous()).cuda(),Variable(y.contiguous()).cuda()
            correct = 0
            #x, y =  Variable(torch.from_numpy(feat[:,:,:])).cuda(), Variable(torch.from_numpy(label[:,:]).long()).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
                else Variable(hidden.data)

            optimizer.zero_grad()
            output, hidden = model(x, hidden,length)
            #loss = compute_loss(output, y, length)
            _,predict = torch.max(output,1)
            predict_data = ((predict.data).cpu().numpy())
            correct = np.sum(predict_data == yt.reshape(-1))
            
            loss= CELOSS(output,y)
            #assert x.size(0) == batch_size
            loss.backward()
            optimizer.step()

            '''
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            for p in model.parameters():
                if p.requires_grad:
                    if args.weight_decay > 0:
                        p.data.mul_(1.0-args.weight_decay)
                    p.data.add_(-lr, p.grad.data)
            '''

            if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
                sys.exit(0)
                return

            total_loss += loss.data[0] * sum(length) 
            running_acc += correct
            total_frame += sum(length)
            i+=1
            if i%5000 == 0:
                for p in model.parameters():
                    sys.stdout.write("maxwts={},minwts={}\n".format(torch.max(p).data[0],torch.min(p).data[0]))
                sys.stdout.flush()
            if i%10 == 0:
                sys.stdout.write("train: time:{}, Epoch={},trbatch={},loss={:.4f},tracc={:.4f}, batchacc={:.4f}, correct={}, total={}\n".format(datetime.now(),epoch,i,total_loss/total_frame,\
                        running_acc*1.0/total_frame, float(correct)/sum(length), correct, sum(length)))
                sys.stdout.flush()

    return total_loss/total_frame,running_acc*1.0/total_frame

def eval_model(epoch,model, valid_reader):
    model.eval()
    args = model.args
    valid_reader.initialize_read(True)
    batch_size = args.batch_size
    total_loss = 0.0
    #criterion = nn.CrossEntropyLoss(size_average=True,ignore_index=-1)
    hidden = model.init_hidden(batch_size)
    i=0
    total_frame=0
    cvacc=0
    while True:
        feat,label,length = valid_reader.load_next_nstreams()
        if length is None or label.shape[0]<args.batch_size:
            break
        else:
            xt=np.copy(np.transpose(feat,(1,0,2)))
            yt=np.copy(np.transpose(label,(1,0)))
            x,y = torch.from_numpy(xt),torch.from_numpy(yt).long()
            x, y = Variable(x.contiguous()).cuda(),Variable(y.contiguous()).cuda()
            correct = 0
            #x, y =  Variable(torch.from_numpy(feat[:,:,:])).cuda(), Variable(torch.from_numpy(label[:,:]).long()).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
                else Variable(hidden.data)

            output, hidden = model(x, hidden,length)
            _,predict = torch.max(output,1)
            predict_data = ((predict.data).cpu().numpy())
            correct = np.sum(predict_data == yt.reshape(-1))
            
            loss = CELOSS(output,y)
            
            total_frame+= sum(length)
            total_loss += loss.data[0]*sum(length)
            cvacc += correct
            i+=1
            if i%10 == 0:
                sys.stdout.write("valid: time:{}, Epoch={},cvbatch={},loss={:.4f},cvacc={:.4f}, batchacc={:.4f}, correct={}, total={}\n".format(datetime.now(),epoch,i,total_loss/total_frame,\
                        cvacc*1.0/total_frame, float(correct)/sum(length), correct, sum(length)))
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    model.print_pnorm()
    sys.stdout.write("\n")

    unchanged = 0
    best_dev = 1e+8
    for epoch in range(args.max_epoch):
        start_time = time.time()
        if args.lr_decay_epoch>0 and epoch>=args.lr_decay_epoch:
            args.lr *= args.lr_decay
        
        train_loss,tracc = train_model(epoch, model, kaldi_io_tr, optimizer)
        cvstart_time = time.time()
        dev_loss,devacc = eval_model(epoch,model, kaldi_io_dev)
        sys.stdout.write("Epoch={}  lr={:.4f}  train_loss={:.4f}  dev_loss={:.4f}  tracc={:.4f}  validacc={:.4f}"
                "\t[{:.4f}m]\t[{:.4f}m]\n".format(
            epoch,
            args.lr,
            train_loss,
            dev_loss,
            tracc,
            devacc,
            (cvstart_time-start_time)/60.0,(time.time()-cvstart_time)/60.0
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
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=175)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--clip_grad", type=float, default=5)

    args = argparser.parse_args()
    print (args)
    main(args)
