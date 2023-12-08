# AML Project README

## Python version and Packages
The python version used for this project was `Python 3.9.18`. The required packages for this project can be installed using the `requirements.txt` file. Please use the below code to install all the packages.

`pip install -r requirements.txt`

## Required datasets
The datasets for this project can be synthetically generated using the script `01_new_data_generate.ipynb`. It is a python notebook and each individual cell needs to be executed to generate the 3 datasets used in this project. 

The data will be generated in the below folder path. (Note: The folder path is relative to the folder that you will save the scripts to)
`data folder path: '../data/new_data/data{data_version}"`

(Note: `data_version` indicates a variable it is 1 , 2 or 3 indicating each synthetic data you generate.)

The dataset household power consumption can be downloaded from `https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set/`

(Note: Please download the file and follow the folder structure below, place the file in the `household_power` folder with file name `household_power_consumption.txt`)

## Code instructions and folder structure
The code for each model created are stored in separate folders and the structure for the same is below, with a short description for each folder and file.

```
├── src
│   ├── hyperparameter_search (Folder with the hyperparameter search results)
│   ├── model_data (Folder with models during hyperparamter search)
│   ├── models (Folder with the best models after hyperparamter search and early stopping)
│   ├── results (Folder with all the results)
│   ├── 01.1_generate_household_train_val_test_sets.ipynb (Script to generate train, val and test sets from household power dataset)
│   ├── 01_new_data_generate.ipynb (Script to generate synthetic data)
│   ├── 02_new_data_model_epistemic_BNN.py (Script to train the Monte carlo drop out and deep ensemble models using the best hyperparameters - for synthetic dataset)
│   ├── 03_hp_search_bnn_models_mcdo.py (Script to get the best hyperparamters for Monte carlo drop out model)
│   ├── 03_hp_search_bnn_models_mcdo.ipynb (Python notebook for the above file)
│   ├── 04_hp_search_bnn_models_deep_ensemble.py (Script to get the best hyperparamters for deep ensemble model)
│   ├── 04_hp_search_bnn_models_deep_ensemble.ipynb (Python notebook for the above file)
│   ├── 06_household_model_epistermic_BNN.py (Script to train the Monte carlo drop out and deep ensemble models using the best hyperparameters - for power consumption dataset)
│   ├── 06_household_model_epistermic_BNN.ipynb (Python notebook for the above file)
|   ├── requirements.txt (Requirements file with all the required packages)
├── data (Folder containing all the data)
│   ├── new_data
│   ├── ├── data1  (Folder with data for 1st synthetic data contains - train, validation and test sets)
│   ├── ├── data2  (Folder with data for 2nd synthetic data contains - train, validation and test sets)
│   ├── ├── data3  (Folder with data for 3rd synthetic data contains - train, validation and test sets)
│   ├── household_power 
│   ├── ├── household_power_consumption.txt (Please place the power consumption dataset here )

```

The python files can be executed using the below code (for synthetic dataset): 
`python <file_name> --dv <data_version>`

The python files for household power consumption data can be executed using the below code: 
`python <file_name>`

Note:
1) `data_version` can be a value 1, 2 or 3.
2) Please follow the above folder structure. 
3) There might be some issues in executing the hyperparamter search code, the issue will arise for the path mentioned in line 278 .
    `local_dir= f'/home/sgupta/WORK/Triplevel_transformer_model/baselines/hyperparameter_search'`
    In above line please change the local_dir to you path to hyperparameter search folder.

## Results
The results for the code will be generated in the `results` folder. 
The expected output for this project are visualization plots and they are all present in the above mentioned folder. Further, we also generate the MSE, RMSE and MAE scores and store it in the results folder. 



