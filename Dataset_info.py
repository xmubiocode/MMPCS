
import pandas as pd
import Config
import os


# 我希望写一个类，里面放的都是配置信息，这样就不用每次都去修改代码了
# 结构为 DatasetName:
#         type: "classification" or "regression"
#         smiles_col: smiles所在的列名
#         label_col: label所在的列名



class DatasetInfo:
    def __init__(self, dataset_name):
        # 数据集名字
        self.dataset_name = dataset_name
        # 回归任务还是分类任务
        self.type = None
        # 数据集文件名
        self.filename = None
        # 分类任务的类别数
        # 如果是回归任务就是 1
        self.class_num = None
        # 这个还没想好怎么搞, 因为一个数据集可能有多个任务
        self.task_num = None
        # smiles所在的列名
        self.smiles_col = None
        # label所在的列名
        self.label_col = None
        # 分割方式
        self.split_type = None
        self.load_info()
        if self.type == "regression":
            self.class_num = 1
    
    def load_info(self):
        # 已设计好
        if self.dataset_name == "BBBP":
            # 要去掉 num 和 name
            self.type = "classification"
            self.tasks_num = 1
            self.filename = "BBBP.csv"
            self.smiles_col = "smiles"

            self.split_type = "Scaffold"
            self.eval_metrics = "ROC_AUC"
        # 已设计好
        elif self.dataset_name == "Tox21":
            # 要去掉一个 mol_id
            self.type = "classification"
            self.tasks_num = 12
            self.filename = "tox21.csv"
            self.smiles_col = "smiles"

            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        elif self.dataset_name == "ToxCast":
            # 没有要去掉的
            self.type = "classification"
            self.tasks_num = 617
            self.filename = "toxcast_data.csv"
            self.smiles_col = "smiles"
            
            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        elif self.dataset_name == "SIDER":
            # 没有要去掉的
            self.type = "classification"
            self.tasks_num = 27
            self.filename = "sider.csv"
            self.smiles_col = "smiles"
            
            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        # 已设计好
        elif self.dataset_name == "ClinTox":
            # 没有要去掉的
            self.type = "classification"
            self.tasks_num = 2
            self.filename = "clintox.csv"
            self.smiles_col = "smiles"
            
            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"
        
        elif self.dataset_name == "MUV":
            # 需要去掉 mol_id
            self.type = "classification"
            self.tasks_num = 17
            self.filename = "muv.csv"
            self.smiles_col = "smiles"
            
            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        elif self.dataset_name == "HIV":
            # 需要去掉 activity
            self.type = "classification"
            self.tasks_num = 1
            self.filename = "HIV.csv"
            self.smiles_col = "smiles"
            
            self.split_type = "Scaffold"
            self.eval_metrics = "ROC_AUC"
        
        # 已设计好
        elif self.dataset_name == "BACE":
            # 只留下 mol 和 Class 其他全要去掉.
            self.type = "classification"
            self.tasks_num = 1
            self.filename = "bace.csv"
            self.smiles_col = "mol"

            self.split_type = "Scaffold"
            self.eval_metrics = "ROC_AUC"
        
        
        # 没找到文件
        elif self.dataset_name == "PCBA":
            self.type = "classification"
            self.tasks_num = 128
            self.smiles_col = "smiles"

            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"
            # self.label
            # 文件不存在的报错
            raise ValueError("没找到数据集文件")
        
        elif self.dataset_name == "Estrogen":
            self.type = "classification"
            self.tasks_num = 2
            self.filename = "estrogen.csv"
            self.smiles_col = "smiles"

            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        elif self.dataset_name == "MetStab":
            self.type = "classification"
            self.tasks_num = 2
            self.filename = "metstab.csv"
            self.smiles_col = "smiles"

            self.split_type = "Rondom"
            self.eval_metrics = "ROC_AUC"

        # 以下五个是回归任务
        elif self.dataset_name == "ESOL":
            self.filename = "delaney-processed.csv"
            self.type = "regression"
            self.smiles_col = "smiles"
            self.label_col = "measured log solubility in mols per litre"

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"

        elif self.dataset_name == "FreeSolv":
            self.filename = "SAMPL.csv"
            self.type = "regression"
            self.smiles_col = "smiles"
            self.label_col = "expt"

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"

        elif self.dataset_name == "Lipo":
            self.filename = "Lipophilicity.csv"
            self.type = "regression"
            self.smiles_col = "smiles"
            self.label_col = "exp"

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"

        elif self.dataset_name == "Malaria":
            self.filename = "malaria-processed.csv"
            self.type = "regression"
            self.smiles_col = "smiles"
            self.label_col = "activity"

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"

        elif self.dataset_name == "cep":
            self.filename = "cep-processed.csv"
            self.type = "regression"
            self.smiles_col = "smiles"
            self.label_col = "PCE"

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"
        else:
            # 报错
            raise ValueError("数据集不存在")

# 辅助任务
# sa
# QED
# logP

# 数据集加载辅助类
class DatasetLoadHelper:
    def __init__(self, datasetinfo):
        self.datasetinfo = datasetinfo
        self.dataset_name = self.datasetinfo.dataset_name
        self.file_path = os.path.join(Config.FINETUNE_DATASET_BASEPATH, self.datasetinfo.filename)
        
        self.smiles_list = self.get_smiles()

        # self.labels_list = None
        self.task_num = None
        if self.datasetinfo.type == "classification":
            self.task_num = self.datasetinfo.tasks_num
        self.tasks_labels_list = None
        self.tasks_class_num = None

        self.load_dataset()
    
    # 我希望对每一个数据集都单独写一个函数，根据数据集名字来加载数据集·
    def get_smiles(self):
        return pd.read_csv(self.file_path, index_col=None)[self.datasetinfo.smiles_col].tolist()
    
    def load_dataset(self):
        if self.datasetinfo.type == "regression":
            self.tasks_labels_list = pd.read_csv(self.file_path, index_col=None)[self.datasetinfo.label_col].tolist()
        elif self.datasetinfo.type == "classification":
            if self.datasetinfo.dataset_name == "BBBP":
                return self.load_BBBP()
            elif self.datasetinfo.dataset_name == "Tox21":
                return self.load_Tox21()
            elif self.datasetinfo.dataset_name == "ToxCast":
                return self.load_ToxCast()
            elif self.datasetinfo.dataset_name == "SIDER":
                return self.load_SIDER()
            elif self.datasetinfo.dataset_name == "ClinTox":
                return self.load_ClinTox()
            elif self.datasetinfo.dataset_name == "MUV":
                return self.load_MUV()
            elif self.datasetinfo.dataset_name == "HIV":
                return self.load_HIV()
            elif self.datasetinfo.dataset_name == "BACE":
                return self.load_BACE()
            elif self.datasetinfo.dataset_name == "PCBA":
                pass
            elif self.datasetinfo.dataset_name == "Estrogen":
                return self.load_Estrogen()
            elif self.datasetinfo.dataset_name == "MetStab":
                return self.load_MetStab()
            else:
                raise ValueError("数据集不存在")
    
    
    # 获取任务信息
    def get_task_info(self, df):
        # 获取df 每个列的名字
        tasks_name = self.get_task_name(df)

        # 获取每个任务的labels 列表
        tasks_labels_list = self.get_task_labels(df)
        # print(tasks_labels_list)

        tasks_class_num, vaild_smaples_num = self.get_task_class_num(tasks_labels_list)
        print("每个任务的类的数量", tasks_class_num)
        print("每个任务的有效样本数量", vaild_smaples_num)

        if len(tasks_name) != self.datasetinfo.tasks_num:
            raise ValueError(f"任务数量异常, 配置为{self.datasetinfo.tasks_num}, 识别到{len(tasks_name)}")

        self.tasks_labels_list = tasks_labels_list
        self.tasks_class_num = tasks_class_num
        
        
        
    # 获得每个任务的名字
    def get_task_name(self, df):
        tasks_name = []
        for col in df.columns:
            print(col)
            tasks_name.append(col)
        return tasks_name
    
    # 获取每个任务的labels 列表
    def get_task_labels(self, df):
        tasks_labels_list = []
        
        for col in df.columns:
            # 如果不是 Nan 就转化为整数，如果是nan就保留
            filter_labels_list = [int(x) if not pd.isna(x) else x for x in df[col].tolist()]
            tasks_labels_list.append(filter_labels_list)
        return tasks_labels_list
    
    # 获得每个任务的类的数量
    def get_task_class_num(self, tasks_labels_list):
        tasks_class_num = []
        vaild_smaples_num = []
        for labels_list in tasks_labels_list:
            # 如果是 NaN 就跳过
            filter_labels_list = [x for x in labels_list if not pd.isna(x)]
            tasks_class_num.append(len(set(filter_labels_list)))
            vaild_smaples_num.append(len(filter_labels_list))
        return tasks_class_num, vaild_smaples_num
    
    def load_BBBP(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["num", "name", "smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)
    
    def load_Tox21(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["mol_id", "smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)
    
    def load_ToxCast(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)

    def load_SIDER(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)

    def load_ClinTox(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)

    def load_MUV(self): 
        df = pd.read_csv(self.file_path, index_col=None).drop(["mol_id", "smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)
    
    def load_HIV(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles", "activity"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)
    
    def load_BACE(self):
        # 这里只选择 Class 列
        df = pd.read_csv(self.file_path, index_col=None, usecols=["Class"])
        
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)

    def load_Estrogen(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)
    
    def load_MetStab(self):
        df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
        tasks_num = df.shape[1]
        print("tasks_num:", tasks_num)
        self.get_task_info(df)


if __name__ == "__main__":
    datasetinfo = DatasetInfo("BACE")
    DatasetLoadHelper(datasetinfo).labels