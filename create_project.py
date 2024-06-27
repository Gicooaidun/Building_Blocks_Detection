from utils.project_manager import create_project


""" Params
    "name": Name of project, should start with the name of the model combined with a unique part. Eg: "UNet_original_bs16" 
    "description": Short description of what makes the model special. Eg: "batchsize 16, otherwise default params"
    "model": Model with input params. Eg:
        "type": UNet_original,
        "params": {
            "n_channels": 3,   
            "n_classes": 1 
            "bilinear": False  
        }
    },
    "w_size": Window size for batches; usually 512
    "data_aug": indication whether to use data augmentation or not; usually False
    "data_aug_mode": augmentation mode; one of the following Options: None, "ctr", "rot", "aff", "ctr+rot", "ctr+aff"; usually None
    "data_dilation": ndication whether to use data dilation or not; usually False
    "n_epochs": number of epochs for training;
    "batch_size": batch size; usually 8
    "criterion": criterion to use; one of the following options:"dice", "bce", "dice+bce"; usually "dice+bce"
    "learning_rate": learning rate; usually 1e-5
    "weight_decay": weigth decay (L1 norm); usually 1e-5
    "gradient_clipping": gradient clipping; usually 1.0
    "amp": mixed precision mode for faster calculation; usually False !seems usually to slow down calculations instead of speed them up!
    "sigmoid_threshold":thershold to use for class assignment; usually 0.5
    },
"""

# define project params
project_list = [
    {
        "name": "project_name",
        "description": "project_description",
        "model": {
            "type": "UNet",
            "params": {
                "n_channels": 3,
                "n_classes": 1,
                "bilinear": False
            }
        },
        # data loader
        "w_size": 512,
        "data_aug": True,
        "data_aug_mode": 'ctr+rot+tps',    
        "data_dilation": False,
        # train
        "n_epochs": 20,
        "batch_size": 8,
        "criterion": "dice+bce",
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "gradient_clipping": 1.0,
        "amp": False,
        # evaluation
        "sigmoid_threshold": 0.5
    }
]

# created project folder and saved project params in .json file
for project in project_list:
    create_project(project)