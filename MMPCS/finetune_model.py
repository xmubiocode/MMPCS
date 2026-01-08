from Model import MyModel, AutoEncoder, FusionAE
import torch.nn as nn
import torch
import os
from torch_geometric.nn import global_mean_pool


class FineTuneModelClassifier(nn.Module):
    def __init__(self, datasetloadhelper, device="cpu", is_multi_gpu=False):
        super(FineTuneModelClassifier, self).__init__()
        self.device = device
        self.model = MyModel(device)
        self.ae_smiles_to_graph = AutoEncoder()
        self.ae_graph_to_smiles = AutoEncoder()
        self.fusion_ae_to_smiles = FusionAE()
        self.fusion_ae_to_graph = FusionAE()

        self.classifiers = nn.ModuleList()
        for i in range(datasetloadhelper.task_num):
            self.classifiers.append(nn.Sequential(
                nn.Linear(128*3, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, datasetloadhelper.tasks_class_num[i]),
            ))
        self.classifiers_tf = nn.ModuleList()
        for i in range(datasetloadhelper.task_num):
            self.classifiers_tf.append(nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, datasetloadhelper.tasks_class_num[i]),
            ))
        
        self.classifiers_gnn = nn.ModuleList()
        for i in range(datasetloadhelper.task_num):
            self.classifiers_gnn.append(nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, datasetloadhelper.tasks_class_num[i]),
            ))

    def forward(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        
        shared_smiles_rep = smiles_rep[:,:128]
        shared_graph_rep = graph_rep[:,:128]
        loss_ae_smiles_to_graph = self.ae_smiles_to_graph(shared_smiles_rep, shared_graph_rep)
        loss_ae_graph_to_smiles = self.ae_graph_to_smiles(shared_graph_rep, shared_smiles_rep)

        fusion_rep_1 = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)
        fusion_rep_2 = torch.cat((smiles_rep[:,128:], graph_rep[:,:128], graph_rep[:,128:]), 1)

        loss_fu_smiles = self.fusion_ae_to_smiles(fusion_rep_1, smiles_rep)
        loss_fu_graph  = self.fusion_ae_to_graph(fusion_rep_2, graph_rep)

        recon_loss = loss_ae_smiles_to_graph + loss_ae_graph_to_smiles + loss_fu_graph + loss_fu_smiles
        rep = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(rep))
        # output = self.classifier(rep)
        return smiles_rep, graph_rep, recon_loss, outputs
    
    def forward_tf(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        outputs = []
        for classifier in self.classifiers_tf:
            outputs.append(classifier(smiles_rep))
        return outputs
    
    def forward_gnn(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        outputs = []
        for classifier in self.classifiers_gnn:
            outputs.append(classifier(graph_rep))
        return outputs
    
    def load_model(self, model_dir, index):
        model_path = os.path.join(model_dir, f"model_pretrain_{index}.pth")
        ae_smiles_to_graph_path = os.path.join(model_dir, f"ae_smiles_to_graph_{index}.pth")
        ae_graph_to_smiles_path = os.path.join(model_dir, f"ae_graph_to_smiles_{index}.pth")
        fusion_ae_to_graph_path = os.path.join(model_dir, f"fusion_ae_to_graph_{index}.pth")
        fusion_ae_to_smiles_path = os.path.join(model_dir, f"fusion_ae_to_smiles_{index}.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.ae_smiles_to_graph.load_state_dict(torch.load(ae_smiles_to_graph_path))
        self.ae_graph_to_smiles.load_state_dict(torch.load(ae_graph_to_smiles_path))
        # self.fusion_ae_to_graph.load_state_dict(torch.load(fusion_ae_to_graph_path))
        # self.fusion_ae_to_smiles.load_state_dict(torch.load(fusion_ae_to_smiles_path))


class FineTuneModelClassifierWoAe(nn.Module):
    def __init__(self, datasetloadhelper, device="cpu", is_multi_gpu=False):
        super(FineTuneModelClassifierWoAe, self).__init__()
        self.device = device
        self.model = MyModel(device)

        self.classifiers = nn.ModuleList()
        for i in range(datasetloadhelper.task_num):
            self.classifiers.append(nn.Sequential(
                nn.Linear(128*3, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, datasetloadhelper.tasks_class_num[i]),
            ))

    def forward(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        
        shared_smiles_rep = smiles_rep[:,:128]
        shared_graph_rep = graph_rep[:,:128]

        rep = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(rep))
        # output = self.classifier(rep)
        return smiles_rep, graph_rep, outputs
    
    def load_model(self, model_dir, index):
        model_path = os.path.join(model_dir, f"model_pretrain_{index}.pth")
        self.model.load_state_dict(torch.load(model_path))

class FineTuneModelRegression(nn.Module):
    def __init__(self, device="cpu"):
        super(FineTuneModelRegression, self).__init__()
        self.device = device
        self.model = MyModel(device)
        self.ae_smiles_to_graph = AutoEncoder()
        self.ae_graph_to_smiles = AutoEncoder()
        self.fusion_ae_to_smiles = FusionAE()
        self.fusion_ae_to_graph = FusionAE()
        # 让 model 关闭梯度
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.regression = nn.Sequential(
                nn.Linear(128*3, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )
        
        self.regression_tf = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )
        
        self.regression_gnn = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        
        shared_smiles_rep = smiles_rep[:,:128]
        shared_graph_rep = graph_rep[:,:128]
        loss_ae_smiles_to_graph = self.ae_smiles_to_graph(shared_smiles_rep, shared_graph_rep)
        loss_ae_graph_to_smiles = self.ae_graph_to_smiles(shared_graph_rep, shared_smiles_rep)

        fusion_rep_1 = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)
        fusion_rep_2 = torch.cat((smiles_rep[:,128:], graph_rep[:,:128], graph_rep[:,128:]), 1)

        loss_fu_smiles = self.fusion_ae_to_smiles(fusion_rep_1, smiles_rep)
        loss_fu_graph  = self.fusion_ae_to_graph(fusion_rep_2, graph_rep)

        recon_loss = loss_ae_smiles_to_graph + loss_ae_graph_to_smiles + loss_fu_graph + loss_fu_smiles
        rep = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)

        output = self.regression(rep)

        return smiles_rep, graph_rep, recon_loss, output
    
    def forward_tf(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        output = self.regression_tf(smiles_rep)
        return output
    
    def forward_gnn(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        output = self.regression_gnn(graph_rep)
        return output
    
    def load_model(self, model_dir, index):
        model_path = os.path.join(model_dir, f"model_pretrain_{index}.pth")
        ae_smiles_to_graph_path = os.path.join(model_dir, f"ae_smiles_to_graph_{index}.pth")
        ae_graph_to_smiles_path = os.path.join(model_dir, f"ae_graph_to_smiles_{index}.pth")
        fusion_ae_to_graph_path = os.path.join(model_dir, f"fusion_ae_to_graph_{index}.pth")
        fusion_ae_to_smiles_path = os.path.join(model_dir, f"fusion_ae_to_smiles_{index}.pth")
        self.model.load_state_dict(torch.load(model_path))
        # 冻结model
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.ae_smiles_to_graph.load_state_dict(torch.load(ae_smiles_to_graph_path))
        self.ae_graph_to_smiles.load_state_dict(torch.load(ae_graph_to_smiles_path))
        self.fusion_ae_to_graph.load_state_dict(torch.load(fusion_ae_to_graph_path))
        self.fusion_ae_to_smiles.load_state_dict(torch.load(fusion_ae_to_smiles_path))

class FineTuneModelRegressionWoAe(nn.Module):
    def __init__(self, device="cpu"):
        super(FineTuneModelRegressionWoAe, self).__init__()
        self.device = device
        self.model = MyModel(device)
        # 让 model 关闭梯度
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.regression = nn.Sequential(
                nn.Linear(128*3, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )
        
        self.regression_tf = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )
        
        self.regression_gnn = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        
        shared_smiles_rep = smiles_rep[:,:128]
        shared_graph_rep = graph_rep[:,:128]

        rep = torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)

        output = self.regression(rep)

        return smiles_rep, graph_rep, output
    
    def forward_tf(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        output = self.regression_tf(smiles_rep)
        return output
    
    def forward_gnn(self, smiles, graph):
        smiles_rep, graph_rep = self.model(smiles, graph)
        output = self.regression_gnn(graph_rep)
        return output
    
    def load_model(self, model_dir, index):
        model_path = os.path.join(model_dir, f"model_pretrain_{index}.pth")
        self.model.load_state_dict(torch.load(model_path))
