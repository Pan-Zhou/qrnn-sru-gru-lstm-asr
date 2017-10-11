export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH=../

Train=tr.300h.scp
Valid=sort_cv.scp
trlab=sort_tr.labels
cvlab=sort_cv.labels
statenum=4043


CUDA_VISIBLE_DEVICES=0 python train_am.py \
	--train=$Train \
	--dev=$Valid \
	--trainlab=$trlab \
	--devlab=$cvlab \
	--dropout=0 \
	--rnn_dropout=0 \
	--bias=0 \
	--feadim=40 \
	--hidnum=525 \
	--depth=7 \
	--statenum=$statenum \
	--batch_size=8 \
	--unroll_size=32 \
	--checkpoint \
	--save_folder=300h-sru-dir \
	--lr=0.001 \
	--lr_decay_epoch=12 \
	--clip_grad=20 \
	--max_epoch=40 \
	--lr_decay=0.5 >pz-300h-7sru525.txt 2>&1 &



CUDA_VISIBLE_DEVICES=1 python train_am.py \
	--train=$Train \
	--dev=$Valid \
	--trainlab=$trlab \
	--devlab=$cvlab \
	--dropout=0 \
	--rnn_dropout=0 \
	--lstm \
	--bias=0 \
	--feadim=40 \
	--hidnum=512 \
	--depth=3 \
	--statenum=$statenum \
	--batch_size=8 \
	--unroll_size=32 \
	--checkpoint \
	--save_folder=300h-lstm-dir \
	--lr=0.001 \
	--lr_decay_epoch=4 \
	--clip_grad=20 \
	--max_epoch=12 \
	--lr_decay=0.5 >pz-300h-3lstm512.txt 2>&1 &

