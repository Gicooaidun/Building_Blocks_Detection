import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import datetime
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from utils.dice_score import dice_loss
from utils.smart_data_loader import *
from utils.json_functions import *
from utils.tiling_functions import reconstruct_from_tiles
from utils.coco import get_coco_pano_metric
from utils.postprocessing import postprocessing


class ModelTrainer:
    """
    Model Trainer Class
    """
    def __init__(
        self,
        device,
        name,
        model,
        w_size,
        data_aug,
        data_aug_mode,
        data_dilation,
        n_epochs,
        batch_size,
        criterion,
        learning_rate,
        weight_decay,
        gradient_clipping,
        amp,
        sigmoid_threshold
    ):
        """
        Constructor
        :param device: Device, either 'cuda' or 'cpu'
        :param name: Name of project (string)
        :param model: Model
        :param w_size: Window size
        :param data_aug: Indicates if data augmentation should be applied in smart data loader
        :param data_aug_mode: Indicates the augmentation mode for smart data loader
        :param data_dilation: Indicates if dilation should be applied in smart data loader
        :param n_epochs: Number of epochs to train model.
        :param batch_size: Batch size for training.
        :param criterion = Loss criterion for evaluation. Either "dice" or "bce" or "dice+bce"
        :param learning_rate: Learning rate.
        :param weight_decay: Weight decay.
        :param gradient_clipping: Gradient clipping to avoid too big step size.
        :param amp: Enable mixed precision.
        :param sigmoid_threshold: Threshold value for the assignment of classes from the sigmoid probabilities.
        """
        
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.w_size = w_size
        self.data_aug =  data_aug
        self.data_aug_mode =  data_aug_mode
        self.data_dilation = data_dilation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.amp = amp
        self.sigmoid_threshold = sigmoid_threshold
        self.wandb_project_name = "Building_Block_Segmentation_4.6_Final"
        self.n_epoch = 1
        self.n_step = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = dice_loss()
        


    def train_step(self, train_loader, experiment):
        """
        Train the model for one epoch.
        :param train_loader: train loader
        :param experiment: wandb experiment
        :return train_loss: train loss of epoch
        """
        
        
        # set model to train mode
        self.model.train()

        # initialize train loss
        train_loss = 0

        # number of batches
        n_train = len(train_loader)

        # include progress bar
        with tqdm(total=n_train, desc=f"Train Epoch {self.n_epoch}/{self.n_epochs}", unit='batches') as pbar:
            # iterate over all batches from the training set
            for input, labels, mask, _ in train_loader:
                
                # put data on selected execution device
                input = input.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                labels = labels.to(device=self.device, dtype=torch.long)
                mask = mask.to(device=self.device, dtype=torch.long)

                # set optimizer gradients to zero
                self.optimizer.zero_grad()
                
                with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                    # forward pass
                    output = self.model(input)

                    # apply mask
                    output = output.squeeze(1) * mask.squeeze(1)
                    labels = labels.squeeze(1)

                    # compute and sum up train loss
                    if self.criterion == "dice":
                        loss = self.dice_loss(output, labels)
                    if self.criterion == "bce":
                        loss = self.bce_loss(output, labels.float())
                    if self.criterion == "dice+bce":
                        loss = self.dice_loss(output, labels)
                        loss += self.bce_loss(output, labels.float())
                    train_loss += loss.item()

                # backward pass
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                
                # update optimizer gradients
                self.grad_scaler.step(self.optimizer)

                # update gradient scaler
                self.grad_scaler.update()

                # update progress bar
                pbar.update()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # log to wandb
                experiment.log({
                    'train loss step': loss.item(),
                    'global step': self.n_step
                })

                self.n_step += 1

        # normalize train_loss
        train_loss /= n_train  
        return train_loss
    


    def validation_step(self, val_loader, experiment, n_tiles_val, n_batches_val, size_img_val, gt_path, mask_path, out_path):
        """
        Evaluate model on validation data.
        :param val_loader: validation loader
        :param experiment: wandb experiment
        :param n_tiles_val: number of tiles for validation image
        :param n_batches_val: number of batches for validation image
        :param size_img_val: size of validation image
        :param gt_path: path of validation ground truth image
        :param mask_path: path of validation mask
        :param out_path: path for validation predction output
        :return val_loss: validation loss
        :return val_accuracy: validation accuracy
        :return val_f1: alidation F1 score
        :return pq: validation panoptic quality
        :return sq: validation segmentation quality
        :return rq: validation recognition quality
        """

        # set model to evaluation mode
        self.model.eval()

        # initialize validation step and validation loss
        val_step = 0
        val_loss = 0
        
        # create ampty list of length number of tiles
        pred_tiles = [None] * n_tiles_val

        # no need to calculate gradients in the validation round
        with torch.no_grad():
            # include progress bar

            with tqdm(total=n_batches_val, desc=f"Validation Epoch {self.n_epoch}/{self.n_epochs}", unit='batches') as pbar:
                # iterate over all batches from the validation set
                for input, labels, mask, index in val_loader:

                    # put data on selected execution device
                    input = input.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    labels = labels.to(device=self.device, dtype=torch.long)
                    mask = mask.to(device=self.device, dtype=torch.long)

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # forward pass
                        output = self.model(input)
                        
                        # apply the mask to the prediction
                        output = output.squeeze(1) * mask.squeeze(1)
                        labels = labels.squeeze(1)

                        # compute an sum up validation loss
                        if self.criterion == "dice":
                            loss = self.dice_loss(output, labels)
                        if self.criterion == "bce":
                            loss = self.bce_loss(output, labels.float())
                        if self.criterion == "dice+bce":
                            loss = self.dice_loss(output, labels)
                            loss += self.bce_loss(output, labels.float())
                        val_loss += loss.item()

                        # predict classes
                        batch_pred_probs = F.sigmoid(output).cpu().numpy()
                        batch_pred_labels = np.where(batch_pred_probs > self.sigmoid_threshold, 1, 0)

                    # update progress bar
                    pbar.update()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    
                    # log two tiles per epoch to wandb
                    division_step = n_batches_val // 2
                    if (val_step % division_step) == (division_step // 2):
                        try:
                            experiment.log({
                                'images': wandb.Image(input[0].cpu()),
                                'predictions': wandb.Image((F.sigmoid(output) > self.sigmoid_threshold)[0].float().cpu()),
                                'labels': wandb.Image(labels[0].float().cpu()),
                            })
                            print("images successfully loaded to wandb")
                        except:
                            print("error while loading images to wandb")

                    # store predicted tiles in prediction list at correct index 
                    for batch_index in range(len(index)):
                        tile_index = index[batch_index]
                        pred_tiles[tile_index] = batch_pred_labels[batch_index]

                    val_step += 1


        # reconstruct full image from tiles
        pred_tiles = np.array(pred_tiles)
        pred_labels = reconstruct_from_tiles(pred_tiles, self.w_size, self.w_size//2, size_img_val, np.uint8)

        # postprocessing
        pred_labels = postprocessing(pred_labels)

        # apply mask
        mask = np.array(Image.open(mask_path))/255
        pred_labels = pred_labels * mask

        # save predicted image and log to wandb
        pred_image = (pred_labels * 255).astype(np.uint8)
        result_img = Image.fromarray((pred_image))
        result_img.save(out_path)
        experiment.log({'predicted image': wandb.Image(pred_image)})
        print("predicted image successfully saved and loaded to wandb")
    
        # calculate loss
        val_loss /= n_batches_val

        # calculate accuracy and f1 
        pred_labels_flat = pred_labels.flatten().astype(int)
        true_labels_flat = np.array(Image.open(gt_path)).flatten()
        true_labels_flat = (true_labels_flat/255).astype(int)
        mask_flat = np.array(Image.open(mask_path)).flatten().astype(bool)
        pred_labels_flat = pred_labels_flat[mask_flat] # apply mask
        true_labels_flat = true_labels_flat[mask_flat] # apply mask
        val_accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
        val_f1 = f1_score(true_labels_flat, pred_labels_flat, average='macro')

        # calculate coco panoptic metrics
        pq, sq, rq = get_coco_pano_metric(gt_path, out_path)

        return val_loss, val_accuracy, val_f1, pq, sq, rq
    


    def train_model(
        self,
        in_dir,
        out_checkpoints_dir
    ):
        """
        Train model.
        :param experiment: wandb experiment
        :param n_tiles_val: number of tiles for validation image
        :return train_summary: dict with training metrics (loss) and validation metrics (loss, accuracy, F1, PQ, SQ and RQ)) for each epoch
        """
        
        # set paths
        train_img_path = f"{in_dir}/train/101-INPUT.jpg"
        train_gt_path = f"{in_dir}/train/101-OUTPUT-GT.png"
        train_mask_path = f"{in_dir}/train/101-INPUT-MASK.png"
 
        val_img_path = f"{in_dir}/validation/201-INPUT.jpg"
        val_gt_path = f"{in_dir}/validation/201-OUTPUT-GT.png"
        val_mask_path = f"{in_dir}/validation/201-INPUT-MASK.png"

        # Initialize wandb run
        experiment = wandb.init(project=self.wandb_project_name, name=self.name, reinit=True, resume='allow', anonymous='must')
        experiment.config.update(
            dict(
                w_size=self.w_size,
                data_aug=self.data_aug,
                data_aug_mode=self.data_aug_mode,
                data_dilation=self.data_dilation,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                criterion=self.criterion,
                learning_rate=self.learning_rate,
                gradient_clipping=self.gradient_clipping,
                amp=self.amp,
                sigmoid_threshold=self.sigmoid_threshold
            )
        )

        # load and prepare data for training
        train_data = Data(
            train_img_path,
            train_gt_path,
            train_mask_path,
            self.w_size,
            data_aug=self.data_aug,
            aug_mode=self.data_aug_mode,
            dilation=self.data_dilation
        )
        val_data = Data(
            val_img_path,
            val_gt_path,
            val_mask_path,
            self.w_size,
            data_aug=False,
            aug_mode=None,
            dilation=False
        )

        # create train loader and validation loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # initialize summary list
        train_loss_list = []
        val_loss_list = []
        val_acc_list = []
        val_f1_list = []
        val_pq_list = []
        val_sq_list = []
        val_rq_list = []
        
        # logging
        print(f'{self.name} start training with {self.model.n_channels} input channels and {self.model.n_classes} output classes.')

        for epoch in range(1, self.n_epochs + 1):
            self.n_epoch = epoch

            # set paths
            epoch_checkpoint_path = f"{out_checkpoints_dir}/checkpoint_epoch_{self.n_epoch}.pth"
            epoch_summary_path = f"{out_checkpoints_dir}/summary_epoch_{self.n_epoch}.json"
            epoch_val_pred_path = f"{out_checkpoints_dir}/pred_val_img_epoch_{self.n_epoch}.png"

            # train model for one epoch
            train_loss = self.train_step(train_loader, experiment)

            # validation step
            n_tiles_val = len(val_data)
            n_batches_val = len(val_loader)
            size_img_val = np.array(Image.open(val_img_path)).shape[:2]
            val_loss, val_accuracy, val_f1, val_pq, val_sq, val_rq = self.validation_step(val_loader, experiment, n_tiles_val, n_batches_val, size_img_val, val_gt_path, val_mask_path, epoch_val_pred_path)
            
            # append epoch metrics to list
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)
            val_f1_list.append(val_f1)
            val_pq_list.append(val_pq)
            val_sq_list.append(val_sq)
            val_rq_list.append(val_rq)
            
            # print epoch metrics summary
            print(f"Epoch {self.n_epoch}:")
            print(f"Training: loss = {train_loss}")
            print(f"Validation: loss = {val_loss}, accuracy = {val_accuracy}, f1 = {val_f1}, pq = {val_pq}, sq = {val_sq}, rq = {val_rq}")

            # log epoch metrics to wandb and save them as .json file
            epoch_summary = {
                'epoch': self.n_epoch,
                'train loss': train_loss,
                'validation loss': val_loss,
                'validation accuracy': val_accuracy,
                'validation f1': val_f1,
                'validation pq': val_pq,
                'validation sq': val_sq,
                'validation rq': val_rq,
            }
            experiment.log(epoch_summary)
            save_json(epoch_summary, epoch_summary_path)
  
            # save checkpoin
            state_dict = self.model.state_dict()
            torch.save(state_dict, epoch_checkpoint_path)

            # logging
            print(f'{self.name} checkpoint epoch {self.n_epoch} saved! \n')

            # update epoch counter
            self.n_epoch += 1

        # logging
        print(f'{self.name} training completed')

        # create summary
        train_summary = {
            "train_loss": train_loss_list,
            "validation_loss": val_loss_list,
            "validation_accuracy": val_acc_list,
            "validation_f1": val_f1_list,
            "validation_panoptic_quality": val_pq_list,
            "validation_segmentation_quality": val_sq_list,
            "validation_recognition_quality": val_rq_list
        }

        # Finish wandb run
        experiment.finish()

        return train_summary