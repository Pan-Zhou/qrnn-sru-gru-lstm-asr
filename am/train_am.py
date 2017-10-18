import sys
import os
import argparse
import time
import random
import math
from datetime import datetime
from more_itertools import sort_together

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from rnnmodel import Model

from io_func.kaldi_feat import KaldiReadIn
from io_func.kaldi_io_parallel import KaldiDataReadParallel

def CELOSS(output,label,delay = 0):
    """
    compute cross entropy loss with net output and label
    ouput: linear before softmax,(N*T, dim) tensor
    label: (T,N) tensor, with -1 padded for padding frames
    return: masked average ce loss
    """
    if delay > 0:
        label.data[delay:,:] = label.data[0:-delay,:]
        label.data[0:delay, :] = -1
    _,predict = torch.max(output,1)
    correct = (predict.data == label.view(-1).data).sum()

    #correct = np.sum(predict_data == yt.reshape(-1))
    
    mask = (label>=0)
    output =F.log_softmax(output)
    labselect = label + (label<0).long()
    select = -torch.gather(output,1,labselect.view(-1,1))
    losses = mask.float().cuda().view(-1,1)*select
    loss = torch.sum(losses)/torch.sum(mask.float())
    return loss, correct


def train_model(epoch, model, train_reader, optimizer):
    model.train()
    args = model.args
    
    train_reader.initialize_read(True)
    batch_size = args.batch_size
    lr = args.lr
    delay = args.target_delay

    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    i=0
    running_acc=0
    total_frame=0
    while True:
        #####batch major tensor,(batch,maxT,feadim),(batch,maxT)
        feat,label,length = train_reader.load_next_nstreams()
        if length is None or label.shape[0]<args.batch_size:
            break
        else:
            ###transpose to time major for rnn training
            xt=np.copy(np.transpose(feat,(1,0,2)))
            yt=np.copy(np.transpose(label,(1,0)))
            x,y = torch.from_numpy(xt),torch.from_numpy(yt).long()
            dx, dy = Variable(x).cuda(),Variable(y).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.rnn_type=='lstm' \
                else Variable(hidden.data)

            output, hidden = model(dx, hidden,length)
            
            optimizer.zero_grad()
            loss, correct = CELOSS(output,dy,delay)
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

            length -= delay
            total_loss += loss.data[0] * sum(length) 
            running_acc += correct
            total_frame += sum(length)
            del loss,output,hidden,dx,dy
            i+=1
            if i%100 == 0:
                sys.stdout.write("train: time:{}, Epoch={},trbatch={},loss={:.4f},tracc={:.4f},
                        batchacc={:.4f}, correct={}, total={}\n".format(
                            datetime.now(),epoch,i,total_loss/total_frame,
                            running_acc*1.0/total_frame, 
                            float(correct)/sum(length), correct, sum(length)))
                sys.stdout.flush()

    return total_loss/total_frame,running_acc*1.0/total_frame

def eval_model(epoch,model, valid_reader):
    model.eval()
    args = model.args
    valid_reader.initialize_read(True)
    batch_size = args.batch_size
    delay = args.target_delay
    total_loss = 0.0
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
            dx, dy = Variable(x,volatile=True).cuda(),Variable(y).cuda()
            hidden = model.init_hidden(batch_size)
            hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.rnn_type=='lstm' \
                else Variable(hidden.data)

            output, hidden = model(dx, hidden,length)
            
            loss, correct = CELOSS(output, dy, delay)
            
            length -= delay
            total_frame+= sum(length)
            total_loss += loss.data[0]*sum(length)
            cvacc += correct
            del loss,output,hidden,dx,dy
            i+=1
            if i%50 == 0:
                sys.stdout.write("valid: time:{}, Epoch={},cvbatch={},loss={:.4f},cvacc={:.4f}, 
                        batchacc={:.4f}, correct={}, total={}\n".format(
                            datetime.now(),epoch,i,total_loss/total_frame,
                        cvacc*1.0/total_frame, float(correct)/sum(length), correct, sum(length)))
                sys.stdout.flush()
    avg_loss = total_loss / total_frame
    return avg_loss,cvacc*1.0/total_frame

def main(args):
    train_read_opts={'label':args.trainlab,'lcxt':args.lcxt,'rcxt':args.rcxt,
            'num_streams':args.batch_size,'skip_frame':args.skipframe}
    dev_read_opts={'label':args.devlab,'lcxt':args.lcxt,'rcxt':args.rcxt,
            'num_streams':args.batch_size,'skip_frame':args.skipframe}

    kaldi_io_tr=KaldiDataReadParallel(args.train,train_read_opts)
    kaldi_io_dev=KaldiDataReadParallel(args.dev,dev_read_opts)

    model = Model(args)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
            eps=1e-08, weight_decay=args.weight_decay)
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    model.print_pnorm()
    sys.stdout.write("\n")

    unchanged = 0
    best_dev = 1e+8
    save_folder = args.save_folder
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if os.path.exists(save_folder):
            print('Directory already exists.')
        else:
            raise

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            start_epoch = checkpoint['epoch']
            dev_loss = checkpoint['dev_loss']
            best_dev = checkpoint['best_dev']
            args.lr = optimizer.param_groups[0]['lr']
            print("=> loaded checkpoint '{}', lrate {:.6f}, dev_loss {:.6f},best_dev loss {:.6f},
                    (epoch {})".format(args.resume, args.lr, dev_loss, best_dev, checkpoint['epoch']))
            if dev_loss >best_dev and start_epoch >=args.lr_decay_epoch:
                args.lr *=args.lr_decay
                optimizer.param_groups[0]['lr']=args.lr
                print("decay lrate to {:.6f}".format(args.lr))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.max_epoch):
        start_time = time.time()
        '''
        if args.lr_decay_epoch>0 and epoch>=args.lr_decay_epoch:
            args.lr *= args.lr_decay
            optimizer.param_groups[0]['lr'] = args.lr
            print("adjust learning rate to {:.6f}\n".format(optimizer.param_groups[0]['lr']))
        '''
        train_loss,tracc = train_model(epoch, model, kaldi_io_tr, optimizer)
        
        cvstart_time = time.time()
        dev_loss,devacc = eval_model(epoch,model, kaldi_io_dev)
        sys.stdout.write("Epoch={}  lr={:.4f}  train_loss={:.4f}  dev_loss={:.4f}  tracc={:.4f}  
                validacc={:.4f}\t[{:.4f}m]\t[{:.4f}m]\n".format(
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
        
        #save best model
        if dev_loss < best_dev:
            unchanged = 0
            best_dev = dev_loss
            file_path = os.path.join(save_folder, args.model_path)
            torch.save({'epoch':epoch+1,
                'dev_loss':dev_loss,
                'best_dev':best_dev,
                'state_dict':model.state_dict(),
                'optim_dict':optimizer.state_dict()},file_path)
            print("Find better validated model, saving to {}".format(file_path))
            start_time = time.time()
            sys.stdout.flush()
        else:
            unchanged += 1
            ##adjust lrate
            if args.lr_decay_epoch >0 and epoch >=args.lr_decay_epoch:
                optimizer.param_groups[0]['lr'] *= args.lr_decay
                print("adjust learning rate to {:.6f}\n".format(optimizer.param_groups[0]['lr']))
        
        #save checkpoit
        if args.checkpoint:
            file_path = '{}/am_{}.pth.tar'.format(save_folder, epoch + 1)
            torch.save({'epoch':epoch+1,
                'dev_loss':dev_loss,
                'best_dev':best_dev,
                'state_dict':model.state_dict(),
                'optim_dict':optimizer.state_dict()}, file_path)
            print('Saving checkpoint model to %s' % file_path)
        
        if unchanged >= 5:
            print("dev_loss unimproved for 5 epoches,terminate training...")
            break
        sys.stdout.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--rnn_type", type=str,required=True,help="rnn type,e.g. lstm,gru,sru,qrnn")
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
    argparser.add_argument("--target_delay", type=int, default=0)
    argparser.add_argument("--max_epoch", type=int, default=300)
    argparser.add_argument("--feadim", type=int, default=40)
    argparser.add_argument("--hidnum", type=int, default=512)
    argparser.add_argument("--dropout", type=float, default=0.5,
        help="dropout of word embeddings and softmax output")
    argparser.add_argument("--rnn_dropout", type=float, default=0.2,
        help="dropout of RNN layers")
    argparser.add_argument("--use_tanh",type=int,default=1,help="sru highway gate activation")
    argparser.add_argument("--bias", type=float, default=-3,
        help="intial bias of highway gates")
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=175)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--clip_grad", type=float, default=5)
    argparser.add_argument("--checkpoint", dest="checkpoint", action="store_true",
                                help="Enables checkpoint saving of model")
    argparser.add_argument('--model_path', default='final.pth.tar',
                            help='Location to save best validation model')
    argparser.add_argument('--save_folder', default='models/',
                                help='Location to save epoch models')
    argparser.add_argument('--resume', default='',
                            help='restore from checkpoint model')
    args = argparser.parse_args()
    print (args)
    main(args)
