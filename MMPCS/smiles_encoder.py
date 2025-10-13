import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from GIN import GNN


class SmilesRoBERTaEncoder(nn.Module):
    def __init__(self, device):
        super(SmilesRoBERTaEncoder, self).__init__()
        self.model_path = "roberta"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        self.roberta_model = RobertaModel.from_pretrained(self.model_path)
        # 把 roberta_model 冻结
        # for param in self.roberta_model.parameters():
        #     param.requires_grad = False
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        # self.tokenizer = self.tokenizer.to(self.device)

    def forward(self, smiles):
        # smiles = smiles.to(self.device)
        inputs = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
        ).to(self.device)

        inputs = {k: v for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs['attention_mask']
        outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        smiles_rep = self.fc(last_hidden_state[:, 0, :])
        return smiles_rep