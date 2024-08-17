from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Recall, F1Score ,Precision, ConfusionMatrix
from torchmetrics.classification.accuracy import Accuracy
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
import numpy as np

class DANNModule(LightningModule):
    def __init__(
        self,
        num_classes:2,
        net: torch.nn.Module,
        # classifier:torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        alpha : float,
        beta : float,
        # bert_weighted_path:str,
        # tokenizer:str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        #self.classifier = classifier
        # loss function
        if num_classes==2:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_domain = torch.nn.BCELoss()
        # self.criterion_rec = F.mse_loss()
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        # other metrics
        self.train_recall = Recall(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.train_pre = Precision(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.train_f1 = F1Score(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.train_confusion = ConfusionMatrix(task="binary")
        self.val_recall = Recall(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.val_pre = Precision(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.val_f1 = F1Score(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.test_recall = Recall(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.test_pre = Precision(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.test_f1 = F1Score(task="binary", num_classes=self.hparams.num_classes,average="weighted")
        self.test_confusion = ConfusionMatrix(task="binary")
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, text,image_path,knowledge_e,domain_label):
        #y = self.net(text,image_path)
        domain_y,classification_y,recon_knowledge = self.net(text,image_path,knowledge_e,domain_label)
        #y = self.classifier(x1)
        return domain_y,classification_y,recon_knowledge
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any, dataloader_id):
        if dataloader_id == 0:
            text,image_path, knowledge_e, label,domain_label = batch
            domain_logits,classification_logits,recon_knowledge = self.forward(text,image_path,knowledge_e,domain_label)
            loss_domain = self.criterion_domain(domain_logits, domain_label)
            # 分别计算loss权重
            #pred_all = torch.argmax(classification_logits, dim=1)
            target_all = np.argmax(label.cpu().numpy(),axis=1)
            indices_true = np.where(target_all==1)
            indices_fake = np.where(target_all==0)
            preds_true = classification_logits[indices_true]
            preds_fake = classification_logits[indices_fake]
            target_true = label[indices_true]
            target_fake = label[indices_fake]
            loss_true = self.criterion(preds_true,target_true)
            loss_fake = self.criterion(preds_fake,target_fake)
            loss_classification = 0.8*loss_true+1.2*loss_fake
            # 正常计算其他损失
            #loss_classification = self.criterion(classification_logits, label)
            loss_recon = F.mse_loss(recon_knowledge, knowledge_e, reduction = "mean")
            preds = torch.argmax(classification_logits, dim=1)
            preds = torch.eye(2,dtype=torch.float).to("cuda")[preds]
            #loss = loss_domain+loss_classification
            loss = loss_classification+self.hparams.alpha*loss_domain+self.hparams.beta*loss_recon
            return loss, preds, label
        elif dataloader_id == 1:
            text,image_path, knowledge_e, label,domain_label = batch
            domain_logits,_,recon_knowledge = self.forward(text,image_path,knowledge_e,domain_label)
            loss_domain = self.criterion_domain(domain_logits, domain_label)
            loss_recon = F.mse_loss(recon_knowledge,knowledge_e, reduction= "mean")
            loss = self.hparams.alpha*loss_domain + self.hparams.beta*loss_recon
            return loss

    def training_step(self, batch: Any, batch_idx: int):
        #print(batch)
        #print(batch_idx)
        # print(dataloader_idx)
        source_batch = batch["source"]
        target_batch = batch["target"]
        loss_1, preds, targets = self.model_step(source_batch,dataloader_id=0)
        loss_2 = self.model_step(target_batch,dataloader_id=1)
        loss = loss_1+loss_2
        pred_label = np.argmax(preds.cpu().numpy(),axis=1)
        true_label = np.argmax(targets.cpu().numpy(),axis=1)
        # update and log metrics
        self.train_confusion.update(preds = torch.tensor(pred_label).cuda(), target = torch.tensor(true_label).cuda())
        self.train_recall(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.train_pre(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.train_f1(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1-score", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("train/confusion_matrix", self.train_confusion, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        print("Train confusion_matrix is:",self.train_confusion.compute())
        self.train_confusion.reset()
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch,dataloader_id=0)
        pred_label = np.argmax(preds.cpu().numpy(),axis=1)
        true_label = np.argmax(targets.cpu().numpy(),axis=1)
        # update and log metrics
        self.val_recall(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.val_pre(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.val_f1(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        
        self.val_loss(loss)
        self.val_acc(preds, targets)
        
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1-score", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch,dataloader_id=0)
        #print(preds)
        #print(targets)
        pred_label = np.argmax(preds.cpu().numpy(),axis=1)
        true_label = np.argmax(targets.cpu().numpy(),axis=1)
        #print(pred_label)
        #print(true_label)
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        # self.test_recall(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.test_confusion.update(preds = torch.tensor(pred_label).cuda(), target = torch.tensor(true_label).cuda())
        self.test_recall(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.test_pre(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        self.test_f1(preds = torch.tensor(pred_label), target = torch.tensor(true_label))
        #print("self.test_recall:",self.test_recall(preds = torch.tensor(pred_label), target = torch.tensor(true_label)))
        #print("self.test_pre:",self.test_pre(preds = torch.tensor(pred_label), target = torch.tensor(true_label)))
        #print("self.test_f1:",self.test_f1(preds = torch.tensor(pred_label), target = torch.tensor(true_label)))
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_pre, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1-score", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/confusion", self.test_confusion, on_step=False, on_epoch=True, prog_bar=True)
        # the metrics we add
        #self.log("test/recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        #print("test precision : {}".format(pre))
        #print("test f1_score : {}".format(f1))
    def on_test_epoch_end(self):
        print("Train confusion_matrix is:",self.train_confusion.compute())
        print("Test confusion_matrix is:",self.test_confusion.compute())
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DANNModule(None, None, None)