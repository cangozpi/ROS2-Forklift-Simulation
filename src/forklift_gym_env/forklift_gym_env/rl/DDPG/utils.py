import numpy as np
import torch
import random


def seed_everything(seed):
    """
    Set seed to random, numpy, torch, gym environment
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)



def log_gradients_in_model(model, logger, step, model_name, log_full_detail=False):
    """
    Logs information (grads, means, ...) about the the parameters of the given model to tensorboard.
    Inputs:
        model (torch.nn.Module): model to log its paramters
        logger (SummaryWriter): information is recorded using the passed tensorboard SummaryWriter
        step (int): x-axis value in the plots
        model_name (str): information will be logged under the given model_name in tensorboard
        log_full_detail (bool): if False, just logs norm of the overall grdients. If True, logs more detailed info per weights and biases.
    """
    all_weight_grads = torch.tensor([])
    all_bias_grads = torch.tensor([])
    # Log weights and gradients to Tensorboard
    for name, param in model.named_parameters():
        if "weight" in name: # Model weight
            if log_full_detail:
                logger.add_histogram(model_name+"/"+name+"/", param, step)
                logger.add_histogram(model_name+"/"+name+"/grad", param.grad, step)
                logger.add_scalar(model_name+"/"+name+"/mean", param.mean(), step)
                logger.add_scalar(model_name+"/"+name+"/grad.mean", param.grad.mean(), step)
            if param.grad is not None:
                all_weight_grads = torch.concat([all_weight_grads, param.grad.cpu().reshape(-1)])

        elif "bias" in name: # Model bias
            if log_full_detail:
                logger.add_histogram(model_name+"/"+name+"/", param, step)
                logger.add_histogram(model_name+"/"+name+"/grad", param.grad, step)
                logger.add_scalar(model_name+"/"+name+"/mean", param.mean(), step)
                logger.add_scalar(model_name+"/"+name+"/grad.mean", param.grad.mean(), step)
            if param.grad is not None:
                all_bias_grads = torch.concat([all_bias_grads, param.grad.cpu().reshape(-1)])
        
    # Log norm of all the model grads concatenated together to form one giant vector
    all_weight_grads_norm = torch.norm(all_weight_grads, 2)
    all_bias_grads_norm = torch.norm(all_bias_grads, 2)
    logger.add_scalar(model_name +"/all_weight_grads_norm", all_weight_grads_norm.item(), step)
    logger.add_scalar(model_name +"/all_bias_grads_norm", all_bias_grads_norm.item(), step)


def log_training_losses(loss, logger, step, model_name):
    """
    Logs training losses during the update of the given model to tensorboard.
    Inputs:
        model (torch.nn.Module): model to log its paramters
        loss (Tensor): loss values to log
        logger (SummaryWriter): information is recorded using the passed tensorboard SummaryWriter
        step (int): x-axis value in the plots
        model_name (str): information will be logged under the given model_name in tensorboard
    """
    # Log to Tensorboard
    logger.add_scalar("Loss"+"/"+model_name, loss, step)
