import os
import pandas as pd

# 读取 dataset_zinc 下面所有以.smi 结尾的文件
smi_list = os.listdir("dataset_zinc")
smi_list = [x for x in smi_list if ".smi" in x]

df_all = pd.DataFrame(columns=["smiles","zinc_id"])

# 读取每一个文件只保留smiles
for file_name in smi_list:
    print(file_name)
    temp_df = pd.read_csv(os.path.join("dataset_zinc",file_name), sep=" ")
    df_all = pd.concat([df_all,temp_df], axis=0)
    # break

print(df_all)

df_all.to_csv("zinc10M.csv", index=False)