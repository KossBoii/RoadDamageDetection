# RoadDamageDetection Project

### Step 1: Clone the github file
```bash
git clone https://github.com/KossBoii/RoadDamageDetection.git\
```

### Step 2: Create new Anaconda Environments to run the code:
```bash
conda create --name py3 python=3.7.0
conda activate py3
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html 
pip install scikit-image

conda install -c anaconda svn
mkdir dataset
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new ./dataset/roadstress_new
svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_old ./dataset/roadstress_old
```

### Step 3: Change to the main working directory
```bash
cd detectron2/projects/RoadStress/
```

# Step 4: Train the model
```bash
sbatch running_script/train.sh training-dataset backbone-model
```

# Step 5: Evalute/Inferencing using the trained model
- For inferencing:\
```bash
sbatch running_script/update_predict.sh [list of models' name]
```
- For evaluation:\
```bash
sbatch running_script/evaluate.sh [list of models' name]
```

Notes for both inference/evaluation script: 
- First go into the output file and get models' folder name (Ex:/output/06102020121824/). By specifiying list of models' name, the script will run through just these files
- Without specifying the list of models' name, the script will run through all the folder existing in the output folder
