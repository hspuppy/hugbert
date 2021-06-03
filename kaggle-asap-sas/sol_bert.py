"""
多文本分类任务，采用distilbert并做微调

Training: 
    python sol_bert.py --batch_size 64 --learning_rate 0.00001 --val_check_interval 100
"""
import logging
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import cohen_kappa
from transformers import AutoModel, AutoTokenizer

MODEL = 'distilbert-base-uncased'
N_CLASS = 4
logger = logging.getLogger(__name__)


class SASDataset(Dataset):
    """Short answer scoring dataset"""
    def __init__(self, input_fn):
        self.data = pd.read_csv(input_fn, sep='\t')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.max_len = 128
        
    def __getitem__(self, index):
        text = self.data.EssayText.iat[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.Score1.iat[index], dtype=torch.long),
        } 
    
    def __len__(self):
        return len(self.data)


class BertClassifier(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, early_stop):
        super().__init__()
        self.save_hyperparameters()
        logger.info(f'hyperparameters: \n{self.hparams}')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.vocab_size = self.tokenizer.vocab_size

        self.dense_input_dim = 768
        self.dropout_rate = 0.2

        self.bert = AutoModel.from_pretrained(MODEL)
        self.dense = torch.nn.Linear(self.dense_input_dim, self.dense_input_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.classifier = torch.nn.Linear(self.dense_input_dim, N_CLASS)

    def train_dataloader(self):
        ds = SASDataset(input_fn='./train.tsv')
        return DataLoader(ds, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        ds = SASDataset(input_fn='./val.tsv')
        return DataLoader(ds, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        ds = SASDataset(input_fn='./test.tsv')
        return DataLoader(ds, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        pooler = self.dense(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        scaled_logits = F.softmax(logits, dim=1)
        label = torch.argmax(scaled_logits, dim=1)
        return label, logits

    def training_step(self, batch, batch_idx):
        ids, mask, y = batch['ids'], batch['mask'], batch['targets']
        _, logits = self.forward(ids, mask)
        loss = F.cross_entropy(logits, y)  # softmax, log, nll_loss
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, y = batch['ids'], batch['mask'], batch['targets']
        _, logits = self.forward(ids, mask)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        val_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        kappa = cohen_kappa(y_hat, y, N_CLASS, weights='quadratic')
        self.log_dict({
            'val_loss': loss,
            'val_acc': val_acc,
            'val_kappa': kappa }, 
            prog_bar=True)

    def test_step(self, batch, batch_idx):
        ids, mask, y = batch['ids'], batch['mask'], batch['targets']
        _, logits = self.forward(ids, mask)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        test_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        kappa = cohen_kappa(y_hat, y, N_CLASS, weights='quadratic')
        self.log_dict({
            'test_loss': loss,
            'test_acc': test_acc,
            'test_kappa': kappa
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--early_stop', type=str, default='val_loss')
        return parser


def train():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = BertClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BertClassifier(
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate, 
        early_stop=args.early_stop
        ) 
    print(model)
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer.from_argparse_args(
        args, 
        callbacks=[early_stopping], 
        precision=16,
        gpus=1, 
        max_epochs=30)
    trainer.fit(model) 


def test():
    saved_model_path = './model.pth'  # lightning_logs/version_N/checkpoints/* 最优模型软链接
    model = BertClassifier.load_from_checkpoint(saved_model_path)
    model.eval()
    print(model)
    trainer = Trainer(gpus=1)
    result = trainer.test(model)
    print(result)


if __name__ == '__main__':
    # train()
    test()
