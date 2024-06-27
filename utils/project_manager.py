import os
import json

def load_project(project_path):

    # read .json file
    with open(project_path, "r") as file:
        project = json.load(file)

    return project


def create_project(project):
    project_folder_path = f"projects/{project['name']}"

    # check if project with this name already exists
    if os.path.exists(project_folder_path):
        print(f"Project {project['name']} already exists, please choose other name")

    else:

        # create model folder
        os.makedirs(project_folder_path)

        # save .json file
        project_json_path = f"{project_folder_path}/project.json"
        with open(project_json_path, "w") as file:
            json.dump(project, file)

        print(f"Successfully created project folder and saved .json file for {project['name']}")



# TODO not in use at the moment
def get_default_param(
    name,
    description,
    model,
    # data loader
    w_size=int(512),
    data_aug=False,
    data_aug_mode=None,
    data_dilation=False,
    # training
    n_epochs=int(5),
    batch_size=int(8),
    criterion="dice+bce",
    learning_rate=float(1e-5),
    weight_decay=float(1e-5),
    gradient_clipping=float(1.0),
    amp=False,
    # evaluation
    sigmoid_threshold=float(0.5)  
):
    project = {
        "name": name,
        "description": description,
        "model": model,
        "w_size": w_size,
        "data_aug": data_aug,
        "data_aug_mode": data_aug_mode,   
        "data_dilation": data_dilation,   
        "n_epochs": n_epochs,
        "batch_size": batch_size,         
        "criterion": criterion, 
        "learning_rate": learning_rate,  
        "weight_decay": weight_decay,   
        "gradient_clipping": gradient_clipping, 
        "amp": amp,             
        "sigmoid_threshold": sigmoid_threshold
    }
    return project