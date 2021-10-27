import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingUnit(unittest.TestCase):

  def test_transformer(self):
    """Sequence-to-Sequence Modeling with nn.Transformer and TorchText

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from datetime import datetime
    TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
    time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
    outdir = outdir if not TIME_STR else (outdir + '_' + time_str)
    print(outdir)

    import collections, shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PositionalEncoding(nn.Module):
      """we use sine and cosine functions of different frequencies."""
      def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

      def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    class TransformerModel(nn.Module):

      def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

      def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

      def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

      def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
          device = src.device
          mask = self._generate_square_subsequent_mask(len(src)).to(device)
          self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    import torchtext
    from torchtext.data.utils import get_tokenizer
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root='datasets')
    TEXT.build_vocab(train_txt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batchify(data, bsz):
      data = TEXT.numericalize([data.examples[0].text])
      # Divide the dataset into bsz parts.
      nbatch = data.size(0) // bsz
      # Trim off any extra elements that wouldn't cleanly fit (remainders).
      data = data.narrow(0, 0, nbatch * bsz)
      # Evenly divide the data across the bsz batches.
      data = data.view(bsz, -1).t().contiguous()
      return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)

    bptt = 35
    def get_batch(source, i):
      seq_len = min(bptt, len(source) - 1 - i)
      data = source[i:i + seq_len]
      target = source[i + 1:i + 1 + seq_len].view(-1)
      return data, target

    ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    import time
    def train():
      model.train()  # Turn on the train mode
      total_loss = 0.
      start_time = time.time()
      ntokens = len(TEXT.vocab.stoi)
      for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
          cur_loss = total_loss / log_interval
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d}/{:5d} batches | '
                'lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval,
            cur_loss, math.exp(cur_loss)))
          total_loss = 0
          start_time = time.time()

    def evaluate(eval_model, data_source):
      eval_model.eval()  # Turn on the evaluation mode
      total_loss = 0.
      ntokens = len(TEXT.vocab.stoi)
      with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
          data, targets = get_batch(data_source, i)
          output = eval_model(data)
          output_flat = output.view(-1, ntokens)
          total_loss += len(data) * criterion(output_flat, targets).item()
      return total_loss / (len(data_source) - 1)


    best_val_loss = float("inf")
    epochs = 3  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      train()
      val_loss = evaluate(model, val_data)
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
      print('-' * 89)

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

      scheduler.step()

    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
      test_loss, math.exp(test_loss)))
    print('=' * 89)
