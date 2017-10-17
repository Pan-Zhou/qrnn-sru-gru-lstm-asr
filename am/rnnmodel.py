import torch
import torch.nn as nn
from torch.autograd import Variable
import cuda_functional as MF

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.feadim
        self.n_cell=args.hidnum
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.n_V = args.statenum
        if args.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.n_d, self.n_cell,self.depth,
                dropout = args.rnn_dropout,batch_first=False
            )
        elif args.rnn_type == 'sru':
            self.rnn = MF.SRU(self.n_d, self.n_cell, self.depth,
                dropout = args.rnn_dropout,
                rnn_dropout = args.rnn_dropout,
                use_tanh = args.use_tanh
            )
        elif args.rnn_type == 'gru':
            self.rnn = nn.GRU(self.n_d, self.n_cell, self.depth, 
                    dropout=args.rnn_dropout, batch_first=False)
        elif args.rnn_type == 'qrnn':
            pass
        else:
            print("unsuported rnn type {}".format(args.rnn_type))
            raise

        self.output_layer = nn.Linear(self.n_cell, self.n_V)

        self.init_weights()
        if  args.rnn_type == 'sru':
            self.rnn.set_bias(args.bias)

    def init_weights(self):
        val_range =0.05 
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, x, hidden,lens):
        ##x=pack_padded_sequence(x, lens, batch_first=True)
        rnnout, hidden = self.rnn(x, hidden)
        ##output,_ = pad_packed_sequence(rnnout, batch_first=True) 
        output = self.drop(rnnout)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_cell).zero_())
        if self.args.rnn_type == 'lstm':
            zeros1 = Variable(weight.new(self.depth,batch_size,self.n_cell).zero_())
            return (zeros, zeros1)
        else:
            return zeros

    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))

