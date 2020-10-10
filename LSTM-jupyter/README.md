# LSTM jupyter notebook

### acknowledgement
 * the LSTM jupyter notebook is adapted from source: https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input/blob/master/LSTM.ipynb
 * the environment02.yml is adopted from matthew's work;
 * md style follows matthew's
 
 ### remark
 * different environment is needed mainly because the code is run on tensorflow version < 2;
 * LSTM model is run for activity recognition (not auslan recognition);
 * once the training and real-time deployment are successfully implemented as python (not ipynb) for this model, shall modify for AUSLAN;

## set-up
0. install the jupyter notebook;
```
conda install -c conda-forge jupyterlab or pip install jupyterlab
```
1. go to the directory where "environment02.yml" resides;
2. create and activate the conda environment;
```
conda env create -f environment.yml
```
3. activate the environment;
```
conda activate tfx1_gpu
```
4. once activated; activate the jupyter notebook;
```
jupyter notebook
```
5. find LSTM.ipynb
6. once everything is done, deactivate the conda environment;
```
conda deactivate
```

## preamble
0. before running the notebook;
1. check the README.md file in the local folder: 'dataset';
2. make sure you have the txt documents mentioned in README before running the notebook;
3. i have placed the real dataset in the local c-drive in the VM as the files are too large for github;
4. check Folder: "LSTM-dataset"; path = "C:\CAPSTONE\capstone2020\LSTM-dataset"
5. then in the notebook; second Cell: "Preparing dataset", change the "DATASET_PATH" to the location where the txt documents reside in your local workspace;
6. then you could play around the model in jupyter notebook (make sure the kernel starts at a clean state);

## remark 02
0. i have attached the expected output when the whole kernel (notebook) is run;
1. check "LSTM_jupyter_results"; ignore page 13 to 121;

## warning
0. the inference when run on the VM took approximately 120 minutes;
1. if impatient or just wanna get the feel, we could just set any positive number of iterations; go to notebook-cell-8-under-"train the network", change "training_iters";
```
while step*batch_size <= training_iters:
```

## to-do
0. implement call-back to save and restore the trained (during-training) model as a safeguard when perform the full training;
1. further abstract the LSTM code for the deployment stage;
