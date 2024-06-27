import os
from train import *
from test import *
from vectorization import *
from utils.project_manager import load_project
from utils.json_functions import *
from utils.random_seed import set_seed
from utils.get_model import get_model



###########################################################
###                      SETTINGS                       ###
###########################################################

# general settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(52)



###########################################################
###                        TRAIN                        ###
###########################################################

def train (project_name):
    '''
    :param project_name: [str] name of project.
    '''

    # set paths and create folders
    project_dir = f"projects/{project_name}"
    data_dir = "data"
    checkpoints_dir = f"{project_dir}/checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    train_summary_path = f"{project_dir}/train_summary.json"

    # load project
    project = load_project(f"{project_dir}/project.json")

    # get train parameters
    train_param = {key: project[key] for key in [
        "name",
        "w_size",
        "data_aug",
        "data_aug_mode",
        "data_dilation",
        "n_epochs",
        "batch_size",
        "criterion",
        "learning_rate",
        "weight_decay",
        "gradient_clipping",
        "amp",
        "sigmoid_threshold"
    ]}

    # get model
    model = get_model(project["model"])

    # initialize model trainer and train model
    model_trainer = ModelTrainer(device=DEVICE, model=model, **train_param)
    train_summary = model_trainer.train_model(data_dir, checkpoints_dir)

    # save train summary dict
    save_json(train_summary, train_summary_path)




###########################################################
###                        TEST                         ###
###########################################################

def test (project_name, epoch=None):
    """
    :param project_name: [str] name of project.
    :param epoch: [int] epoch to use for testing. Default None indicates to use epoch with highest panoptic quality for validation image.
    """

    # set paths
    project_dir = f"projects/{project_name}"
    test_data_dir = "data/test"
    checkpoints_dir = f"{project_dir}/checkpoints"
    train_summary_path = f"{project_dir}/train_summary.json"
    predictions_dir = f"{project_dir}/predictions"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # load project and train summary
    project = load_project(f"{project_dir}/project.json")
    train_summary = read_json(train_summary_path)

    # select epoch with best panoptic quality if no epoch is given
    if epoch is None:
        epoch = np.argmax(np.array(train_summary["validation_panoptic_quality"])) + 1

    # get test parameters
    test_param = {key: project[key] for key in [
        "name",
        "w_size",
        "sigmoid_threshold"
    ]}

    # get model
    model = get_model(project["model"])

    # test model
    model_tester = ModelTester(device=DEVICE, model=model, **test_param)
    test_summary = model_tester.test_model(trained_model_dir=checkpoints_dir, test_data_dir=test_data_dir, predictions_dir=predictions_dir, epoch=epoch)

    # save test summary dict
    test_summary_path = f"{project_dir}/test_summary_epoch_{epoch}.json"
    save_json(test_summary, test_summary_path)



###########################################################
###                    VECTORIZATION                    ###
###########################################################

def vectorization(project_name):
    project_dir = f"projects/{project_name}"
    predictions_dir = f"{project_dir}/predictions"
    vector_dir = f"{project_dir}/vectorization"
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)

    vectorize(predictions_dir, vector_dir)



###########################################################
###                     DEFINE TASKS                    ###
###########################################################

if __name__ == '__main__':

    train("UNet_Final")
    test("UNet_Final")
    vectorization("UNet_Final")
    train("ResUNet_Final")
    test("ResUNet_Final", 14)
    vectorization("ResUNet_Final")
    train("SwinUNet_Final")
    test("SwinUNet_Final")
    vectorization("SwinUNet_Final")