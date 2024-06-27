----- File Structure -----

data – folder with train, validation and test data

models – folder with python files where the UNet, ResUNet and SwinUNet models are specified

projects – folder with the defined projects (models with specified parameters to be trained
including their trained weights, prediction output, vectorization output and performance
metrics if available -> see structure of project folder

utils – folder with python files that provide utility and helper functions

create_project.py – file to create a new project folder with defined project parameters

environment.yml – configuration file of the environment with a list of all dependencies

LICENSE.txt – license under which the software is distributed

main.py – main interface to train model, test model and vectorize building block polygons

train.py – class with functions to train model, includes a training and validation step

test.py – class with functions to test model on all three test images for a specified epoch

vectroization.py – function to vectorize building block predictions


----- Structure of Project Folder -----

Checkpoints – checkpoints of trained model weights as well as predicted validation image and 
summary of training metrics for each epoch

Predictions – predictions of test images

Vectorization – vectorized prediction images, contains the raw and the generalized polygons

project.json – file where the project parameters are specified

test_summary_epoch_XY.json – file where the evaluation metrics (Accuracy, F1, PQ, SQ, RQ) of
the test images for a certain epoch (XY) are stored

train_summary.json – file where the train metrics (train loss, validation loss, validation
accuracy, validation F1, validation PQ, validation SQ, validation RQ) for each epoch are stored


----- Instructions how to use the code -----

1) Create a virtual environment with the required packages specified in the environment.yml file.

2) Specify your model type and the training parameters in the create_project.py file to create a 
new project folder containing the project.json file, or use an existing project (skip this step).

3) Run the train part in main.py  to train your project, the trained weights for each epoch will 
be stored in the checkpoints folder.

4) Run the test part in main.py to test your model (you can specify the epoch yourself or use the 
default settings) and predict the building blocks for all test images, which will be stored in 
the prediction folder.

5) Run the vectorization part in main.py to vectorize your predictions, the output will be stored 
in the vectorization folder.

6) Optimize your model by creating new projects with the adapted parameters.

Note 1: It is possible to combine steps 3)-5) in the same run and for multiple projects.

Note 2: To reproduce the results of the report, run the main.py as it is provided. As the 
projects are already trained, the previous results will be overwritten.s
