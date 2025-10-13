# 🧬 MMPCS: Multi-view Molecular Pretraining Based on Consistency Information and Specific Information
---

# Environment
```bash
conda cteate -n mmpcs python=3.10
conda activate mmpcs

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl

pip install  torch_cluster-1.6.1+pt20cu118-cp310-cp310-linux_x86_64.whl 
pip install  torch_scatter-2.1.1+pt20cu118-cp310-cp310-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt20cu118-cp310-cp310-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt20cu118-cp310-cp310-linux_x86_64.whl 

pip install rdkit==2022.3.5
pip install torch_geometric==2.3.1
pip install transformers==4.33.2
pip install numpy==1.26.1
pip install tabulate==0.9.0
```
# 运行代码

## 📂 数据文件

链接：[https://pan.baidu.com/s/1hUuk3HtASd_LvfiefzsdHA](https://pan.baidu.com/s/1hUuk3HtASd_LvfiefzsdHA)  
提取码：`yqfv`  
> 来自百度网盘超级会员V1的分享，包含：
> - 预训练数据集  
> - 微调数据集  
> - RoBERTa 的预训练模型  

---

## 🚀 预训练

```bash
python main.py
```

---

## 🔧 微调

预训练结束后，会在当前目录下生成一个文件夹，  
记住该文件夹的名称，微调时作为读取预训练模型的输入路径。

示例（以 Estrogen 分类任务为例）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node 8 main_finetune.py
```

---

> 💡 **提示：** 若仅使用单卡微调，可将上述命令简化为：
> ```bash
> python main_finetune.py
> ```

---

这样即可完成从预训练到微调的完整流程。

