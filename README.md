# Mask R-CNN Project (Road Damage Detection)

# Table of Contents
1. [General Info](#general-info)  
2. [Installation](#installation)  
2.1. [HPC](#high-performance-computing-(hpc))  
2.2. [Local Machine](#local-machine)  
2.3. [Google CoLab](#google-colab)
3. [Notes](#notes)

# **General Info**

General Information about the project (In Progress...)

# **Installation**

Intro to Installation (In Progress...)

## **High Performance Computing (HPC)**

<br />  

### **Step 0:** Logging into HPC (Cal Poly Pomona)

- Fill in the form at this [link](https://www.cpp.edu/lrt/hpc/hpc-support.shtml) to request the access to HPC

- Follow the instruction in this [link](https://cpp.service-now.com/ehelp?id=kb_article&sys_id=eb690ad2dbe30410ae3a567b4b9619ef) to setup VPN

- Read the slides in this [link](https://www.cpp.edu/lrt/hpc/hpc-resources.shtml) to go over the basics of HPC

- When connected to CPP's VPN, run the command below and type in the password to access HPC through command line (where username is CPP's username+password credentials)

```bash
ssh -l [username] hpc.cpp.edu
````

*Note: There are 2 folders that a user can access in HPC*

- /home/username/ : **main directory**, where all the packages are installed (very limited storage-wised) 

- /data03/home/username/ : **main working directory**, where to clone the code (no limit in storage)

<br />

### **Step 1:** Clone the GitHub Repository in */data03/home/username*:

```bash
cd /data03/home/username/
git clone https://github.com/KossBoii/RoadDamageDetection.git
cd RoadDamageDetection
```

<br />

### **Step 2:** Create & Setup Anaconda Environment:

Create the Anaconda Virtual Environment
    
```bash
conda create --name py3 python=3.7.0
conda activate py3
pip install opencv-python
pip install scikit-image
```

a) Having Nvidia card (Using GPU to train ~ **faster**):

*Note: Make sure to go to this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) to download and setup CUDA Toolkit 11.3*

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

b) Not Having Nvidia card (Using CPU to train ~ **slower**):

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

<br />

### **Step 3:** Fetch the Dataset:

```bash
conda install -c anaconda svn
mkdir dataset
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new ./dataset/roadstress_new
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_old ./dataset/roadstress_old
```

<br />

### **Step 4:** Train the Models:

```bash
sbatch running_script/train.sh [training-dataset] [backbone-model]
```

<br />

### **Step 5:** Evaluating/Inferencing the Trained Models:

**Inferencing:**

```bash
sbatch running_script/update_predict.sh [list of models' name]
```

**Evaluation:**

```bash
sbatch running_script/evaluate.sh [list of models' name]
```

<br />
<br />

## **Local Machine**

<br />
 
### **Step 1:** Clone the GitHub Repository in the directory you want:

```bash
git clone https://github.com/KossBoii/RoadDamageDetection.git
```

<br />

### **Step 2:** Create & Setup Anaconda Environment:
Create the Anaconda Virtual Environment
    
```bash
conda create --name py3 python=3.7.0
conda activate py3
pip install opencv-python
pip install scikit-image
```

a) Having Nvidia card (Using GPU to train ~ **faster**):

*Note: Make sure to go to this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) to download and setup CUDA Toolkit 11.3*


```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

b) Not Having Nvidia card (Using CPU to train ~ **slower**):

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

<br />

### **Step 3:** Fetch the Dataset:
```bash
conda install -c anaconda svn
mkdir dataset
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new ./dataset/roadstress_new
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_old ./dataset/roadstress_old
```

<br />

### **Step 4:** Train the Models:

```bash
python3 train.py --training-dataset=[dataset_name] --backbone=[backbone_model]
```

<br />

### **Step 5:** Evaluating/Inferencing the Trained Models:

**Inferencing:**

```bash
python3 update_infer.py --config-file "./output/[folder_name]/config.yaml" \
			--dataset "./dataset/" \
		 	--weight "./output/[folder_name]/model_final.pth" \
		 	--output "./prediction/[folder_name]"
```

**Evaluation:**

```bash
python3 evaluate.py --config-file "./output/[folder_name]/config.yaml" \
			--dataset "./dataset/" \
		 	--weight "./output/[folder_name]/model_final.pth" \
		 	--output "./prediction/[folder_name]"
```

<br />
<br />

## **Google CoLab**

... In Progress ...

<br />
<br />

# Notes:

Notes for both inference/evaluation script:

- HPC: 

    - First go into the output file and get models' folder name (Ex:/output/model_3/). By specifiying list of models' name, the script will run through just these files

    - Without specifying the list of models' name, the script will run through all the folder existing in the output folder

- Local Machine:

    - Inference/evaluation script can only be run each trained model separately