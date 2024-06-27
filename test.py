import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.util import view_as_windows
from sklearn.metrics import accuracy_score, f1_score
from utils.tiling_functions import *
from utils.coco import get_coco_pano_metric
from utils.postprocessing import postprocessing


class ModelTester:
    """
    Model Trainer Class
    """
    def __init__(self, device, name, model, w_size, sigmoid_threshold):
        """
        Constructor
        :param device: Device, either 'cuda' or 'cpu'
        :param name: Name of project (string)
        :param model: Model
        :param w_size: Window size
        :param sigmoid_threshold: Threshold value for the assignment of classes from the sigmoid probabilities.
        """
        self.device = device
        self.name = name
        self.model = model.to(device)
        self.w_size = w_size
        self.sigmoid_threshold = sigmoid_threshold

        self.file_numbers = [301, 302, 303]


    def predict_and_evaluate(self, img_path, gt_path, mask_path, out_path):
        """
        :param img_path: Path of input image
        :param gt_path: Path of ground truth image
        :param mask_path: Path of mask
        :param out_path: Path for prediction output
        :return summary: dict with evaluatin metrics (Accuracy, F1, PQ, SQ and RQ) of test prediction
        """
        
        ### prediction ###
        
        # generate image tiles and read 
        tiles = np.array(generate_tiling(img_path, self.w_size))

        # predict tiles
        n_tiles = tiles.shape[0]
        predicted_tiles = []
        with tqdm(total=n_tiles, desc=f"Prediction", unit='tiles') as pbar:
            for tile_image in tiles:

                # preprocessing
                tile_image = np.array(tile_image/255., dtype=np.float32)
                tile_image = tile_image.transpose((2, 0, 1))
                tile_image = torch.from_numpy(tile_image).float().unsqueeze(0)
                tile_image = tile_image.to(self.device)

                # prediction
                with torch.no_grad():
                    output = self.model(tile_image)
                    output = F.sigmoid(output).cpu()
                
                # postprocessing
                predicted_tile = (output[0]).squeeze().numpy()
                predicted_tile = np.where(predicted_tile > self.sigmoid_threshold, 1, 0)

                # add tile to predicted_tiles list
                predicted_tiles.append(predicted_tile)

                pbar.update()

        predicted_tiles = np.array(predicted_tiles)

        # reconstruct full image from tiles
        result = reconstruct_from_tiles(predicted_tiles, self.w_size, self.w_size//2, np.array(Image.open(img_path)).shape[:2], np.uint8)

        # postprocessing
        result = postprocessing(result)

        # apply mask
        mask = np.array(Image.open(mask_path), dtype=np.uint8)/255
        result = (result*mask).astype(np.uint8)

        # save predicted image
        result_img = Image.fromarray((result*255).astype(np.uint8))
        result_img.save(out_path)


        ### evaluation ###

        # read labels and preprocessing
        labels = np.array(Image.open(gt_path), dtype=np.uint8)/255
        true_labels = labels.flatten().astype(int)
        pred_labels = result.flatten().astype(int)

        # calculate accuracy, f1 score and cocopanoptic metrics
        test_accuracy = accuracy_score(true_labels, pred_labels)
        test_f1 = f1_score(true_labels, pred_labels, average='macro')
        pq, sq, rq = get_coco_pano_metric(gt_path, out_path)
        summary = {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "panoptic_quality": pq,
            "segmentation_quality": sq,
            "recognition_quality": rq
        }

        return summary
    


    def test_model (self, trained_model_dir, test_data_dir, predictions_dir, epoch):
        """
        :param trained_model_dir: Directory with checkpoints of trained model
        :param test_data_dir: Directory with test files
        :param predictions_dir: Directory for predictions
        :param epoch: Indicates which epoch checkpoints to use for test prediction
        :return test_summary: dict with evaluatin metrics (Accuracy, F1, PQ, SQ and RQ) of test prediction for each file
        """

        # load trained model
        trained_model_path = f"{trained_model_dir}/checkpoint_epoch_{epoch}.pth"
        state_dict = torch.load(trained_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f'Model {self.name}, Epoch {epoch} loaded for test prediction.')
        
        # set model to evaluation mode
        self.model.eval()

        # predict classes for each input test image
        test_summary = {}
        for file_number in self.file_numbers:

            # set paths
            test_img_path = f'{test_data_dir}/{file_number}-INPUT.jpg'
            test_gt_path = f"{test_data_dir}/{file_number}-OUTPUT-GT.png"
            test_mask_path = f"{test_data_dir}/{file_number}-INPUT-MASK.png"
            out_prediction_path = f"{predictions_dir}/{file_number}_prediction_epoch_{epoch}.png"

            # predict and evaluate
            file_test_summary = self.predict_and_evaluate(
                test_img_path,
                test_gt_path,
                test_mask_path,
                out_prediction_path
            )

            # print test summary
            print(f"File {file_number}:")
            print("Test accuracy: ", file_test_summary["test_accuracy"])
            print("F1 score: ", file_test_summary["test_f1"])
            print("Panoptic quality: ", file_test_summary["panoptic_quality"])
            print("Segmentation quality: ", file_test_summary["segmentation_quality"])
            print("Recognition quality: ", file_test_summary["recognition_quality"])
            
            # apped summary to dict
            test_summary[file_number] = file_test_summary

        return test_summary

