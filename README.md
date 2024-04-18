## End to End Project On NYC Taxi Trip Duration Prediction

### Step 1: Create a `new environment`

```
conda create -p venv python==3.10.14

conda activate venv/
```
### Step 2: Create a `.gitignore file`

```
create the file by right click and include the all the files which you do not need to push to github
```

### Step 3: Create a `requirements.txt` file 
```
pip install -r requirements.txt
```

### Step 4: Create a `setup.py` file 
```
This is to install the entire project as a package. Additionally, write a function to read the packages from requirements.txt
```

### Step5: Create a folder `src` 
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The src folder will include a folder with name components and another one with name pipeline. Include __init__.py also in both of them. 
```
#### Step 5.1 Create a folder `components`

```
Include data_ingestion, data_transformation, model trainer, and __init_.py. These components are to be interconnected in future. 
```
#### Step 5.2 Create a folder called `pipeline`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
``` 

### Step 6: Create a folder called `dataset` 
```
Create a folder called dataset and leave it blank. Data will be pulled from sql and a local copy of that data will be saved as csv in dataset folder.
```