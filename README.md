# Mask R-CNN Project (Road Damage Detection)

## **High Performance Computing (HPC)**

### **Step 0:** Logging into HPC (Cal Poly Pomona)

- Fill in the form at this [link](https://www.cpp.edu/lrt/hpc/hpc-support.shtml) to request the access to HPC

- Follow the instruction in this [link](https://cpp.service-now.com/ehelp?id=kb_article&sys_id=eb690ad2dbe30410ae3a567b4b9619ef) to setup VPN

- Read the slides in this [link](https://www.cpp.edu/lrt/hpc/hpc-resources.shtml) to go over the basics of HPC

- When connected to CPP's VPN, run the command below and type in the password to access HPC through command line (where username is CPP's username+password credentials)

```bash
ssh -l [username] hpc.cpp.edu
```

*Note: There are 2 folders that a user can access in HPC*

- /home/username/ : **main directory**, where all the packages are installed (very limited storage-wised) 

- /data03/home/username/ : **main working directory**, where to clone the code (no limit in storage)

### **Step 1:** Clone the GitHub Repository in */data03/home/username*:

```bash
cd /data03/home/username/
git clone https://github.com/KossBoii/RoadDamageDetection.git
```

### **Step 2:** Create & Setup Anaconda Environment:

- Having Nvidia card (Using GPU to train ~ **faster**):

*Note: Make sure to go to this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) to download and setup CUDA Toolkit 11.3*

```bash
conda create --name py3 python=3.7.0
conda activate py3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python
pip install scikit-image
```

- Not Having Nvidia card (Using CPU to train ~ **slower**):
```bash
conda create --name py3 python=3.7.0
conda activate py3
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python
pip install scikit-image
```




## Local Machine


## Google CoLab





# RoadDamageDetection Project

### Step 1: Clone the github file
```bash
git clone https://github.com/KossBoii/RoadDamageDetection.git
```

### Step 2: Create new Anaconda Environments to run the code:
Note: Make sure to go to this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) to download and setup CUDA Toolkit 11.3
```bash
conda create --name py3 python=3.7.0
conda activate py3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install opencv-python
pip install scikit-image

conda install -c anaconda svn
cd detectron2/projects/RoadStress
mkdir dataset
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new ./dataset/roadstress_new
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_old ./dataset/roadstress_old
```

### Step 3: Change to the main working directory
```bash
cd detectron2/projects/RoadStress/
```

### Step 4: Train the model
```bash
sbatch running_script/train.sh training-dataset backbone-model
```

### Step 5: Evalute/Inferencing using the trained model
- For inferencing:
```bash
sbatch running_script/update_predict.sh [list of models' name]
```
- For evaluation:
```bash
sbatch running_script/evaluate.sh [list of models' name]
```

**Notes for both inference/evaluation script:** 
- First go into the output file and get models' folder name (Ex:/output/06102020121824/). By specifiying list of models' name, the script will run through just these files
- Without specifying the list of models' name, the script will run through all the folder existing in the output folder
