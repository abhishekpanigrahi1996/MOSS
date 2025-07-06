import torch.optim

import engine.abs_engine as abs_engine
import model.vanilla_fusion as fusion_func
import torch.nn as nn
from model.device_check import *
import torch.nn.functional as F
import tool.dynamic as dynamic
import dataset.tweet.dataset as tweet_set
from datasets import load_dataset

class FixHyperfuse(abs_engine.EncoderClassifierEarlyStop):
    def init_dataset(self, cfg_data):
        
        self.train_set = tweet_set.Product("train", cfg_data["num_subsets"], synth=cfg_data["synthetic"])
        self.val_set = tweet_set.Product("val", cfg_data["num_subsets"], synth=cfg_data["synthetic"])
        self.test_set = tweet_set.Product("test", cfg_data["num_subsets"], synth=cfg_data["synthetic"])

        self.batch_size = cfg_data["batch_size"]
        self.val_batch_size = cfg_data["val_batch_size"]
        self.subsets = cfg_data['num_subsets']

    def init_models(self, cfg_train):
        self.loss_std_threshold = 1e-4
        # self.V_enc = nn.Linear(768, cfg_train["hidden_dim"]).to(device)
        # self.T_enc = nn.Linear(768, cfg_train["hidden_dim"]).to(device)

        self.encoder = nn.Sequential(
            nn.Linear(cfg_train["fusion_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        ).to(device)

        if "param" in cfg_train["fusion"]:
            self.fusion = dynamic.import_string(cfg_train["fusion"]["model"])(**cfg_train["fusion"]["param"]).to(device)
        else:
            self.fusion = dynamic.import_string(cfg_train["fusion"]["model"])().to(device)


        self.loss_func = torch.nn.CrossEntropyLoss()

        # init the optimizer
        self.optim = torch.optim.AdamW(
            params=list(self.encoder.parameters()) + list(self.classifier.parameters()) +
                   list(self.fusion.parameters()),
            lr=float(cfg_train['lr']),
            weight_decay=float(cfg_train['weight_decay'])
        )

        self.trained_models = ["encoder", "classifier", "fusion"]

    def forward_pass(self, input_tuple):
        Vv, Tt, Y, V = input_tuple
        
        Vv = torch.stack(Vv)
        Tt = torch.stack(Tt)
        Y = torch.stack(Y)
        # V = torch.stack(V)
        # latent_Vv = self.V_enc(Vv.type(torch.FloatTensor).to(device))
        # latent_Tt = self.T_enc(Tt.type(torch.FloatTensor).to(device))
        # print("v ", Vv.shape)
        # print("t ", Tt.shape)

        return F.softmax(self.classifier(self.encoder(self.fusion(Vv, Tt))), dim=1), \
            Y.type(torch.FloatTensor).to(device)


class FixTensorFusion(FixHyperfuse):
    def forward_pass(self, input_tuple):
        # print(self.encoder)
        Vv, Tt, Y, V = input_tuple
        
        fused = self.fusion([
            Vv.type(torch.FloatTensor).to(device), 
            Tt.type(torch.FloatTensor).to(device)
        ])
        # print("fused: ", fused.shape)
        encoded = self.encoder(fused)
        classified = self.classifier(encoded)
        softmaxed = F.softmax(classified, dim=1)
        return softmaxed.to(device), Y.type(torch.FloatTensor).to(device)
