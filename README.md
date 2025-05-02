# DMD³C: Distilling Monocular Foundation Model for Fine-grained Depth Completion  
**Official Code for the CVPR 2025 Paper**  
**"[CVPR 2025] Distilling Monocular Foundation Model for Fine-grained Depth Completion"**  

[📄 Paper on arXiv](https://arxiv.org/abs/2503.16970)

---

## 🆕 Update Log

- **[2025.04.23]** We have released the **2rd stage training code**! 🎉  
- **[2025.04.11]** We have released the **inference code**! 🎉  

## ✅ To Do

- [ ] 📦 Easy-to-use **data generation pipeline**
- [ ] 🧠 **Checkpoints** trained on a **larger mixed dataset**
- [ ] 🤖 Inference code for **SLAM applications**

> ⚠️ **Note:** We're currently struggling with the response to a journal manuscript revision 📝😅.  
> Thanks for your patience and continued support — we're doing our best to roll out updates as soon as we can! 🙏
---

<div align="center">
  <img width="729" alt="DMD3C Results" src="https://github.com/user-attachments/assets/da4a34ea-0390-418c-8111-22b2096110eb" />
</div>

---

## 🔍 Overview

DMD³C introduces a novel framework for **fine-grained depth completion** by distilling knowledge from **monocular foundation models**. This approach significantly enhances depth estimation accuracy in sparse data, especially in regions without ground-truth supervision.

---
![image](https://github.com/user-attachments/assets/f24eef8e-5dc2-483a-bb70-67671ff5e4e9)


---



## 🚀 Getting Started

### 1. Clone Base Repository

```bash
git clone https://github.com/kakaxi314/BP-Net.git
```

### 2. Copy This Repo into the BP-Net Directory

```bash
cp DMD3C/* BP-Net/
cd BP-Net/DMD3C/
```

### 3. Download Pretrained Checkpoints

- 📥 [Google Drive – Checkpoints] Comming soon... 

### 4. Prepare KITTI Raw Data

Download any sequence from the **KITTI Raw dataset**, which includes:

- Camera intrinsics  
- Velodyne point cloud  
- Image sequences  

Make sure the structure follows the **standard KITTI format**.

### 5. Modify the Sequence in `demo.py` for Inference

Open `demo.py` and go to **line 338**, where you can modify the input sequence path according to your downloaded KITTI data.

```python
# demo.py (Line 338)
sequence = "/path/to/your/kitti/sequence"
```

### 6. Run Inference Demo

```bash
bash demo.sh
```

You will get results like this:

![supp-video 00_00_00-00_00_30](https://github.com/user-attachments/assets/a1412bca-c368-4d19-a081-79eeabaa2901)

### 7. Train on KITTI

Runing monocular depth estimation for all KITTI-raw images. Data structure:
```
├── datas/kitti/raw/
│   ├── 2011_09_26
│   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── image_02
│   │   │   │   ├── data/*.png
│   │   │   │   ├── disp/*.png
│   │   │   ├── image_03
│   │   ├── 2011_09_26_drive_0002_sync.......
```

Where disparity images are stored in gray-scale.

Download pre-trained checkpoitns:
```
wget https://github.com/Sharpiless/DMD3C/releases/download/pretrain-checkpoints/pretrained_mixed_singleview_256.pth
mv pretrained_mixed_singleview_256.pth checkpoints
```

Run metric-finetuning on KITTI dataset:
```
torchrun --nproc_per_node=4 --master_port 4321 train_distill.py \
    gpus=[0,1,2,3] num_workers=1 name=DMD3D_BP_MIXED_ \
    ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
    net=PMP data=KITTI \
    lr=5e-4 train_batch_size=2 test_batch_size=1 \
    sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
    nepoch=30 test_epoch=25 ++net.sbn=true 
```
