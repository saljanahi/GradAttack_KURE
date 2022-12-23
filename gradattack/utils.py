"""Some helper functions for PyTorch, including:
    - parse_args, parse_augmentation: parse for command-line options and arguments
    - get_mean_and_std: calculate the mean and std value of dataset.
    - save_fig: save an image array to file.
    - patch_image: patch a batch of images for better visualization.
    - cross_entropy_for_onehot: cross-entropy loss for soft labels.
"""

import argparse
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import log_softmax, cross_entropy
from functools import reduce


def parse_args():
    parser = argparse.ArgumentParser(description="gradattack training")
    parser.add_argument("--gpuid", default="0", type=int, help="gpu id to use")
    parser.add_argument("--model",
                        default="ResNet18",
                        type=str,
                        help="name of model")
    parser.add_argument("--data",
                        default="CIFAR10",
                        type=str,
                        help="name of dataset")
    parser.add_argument(
        "--results_dir",
        default="./results",
        type=str,
        help="directory to save attack results",
    )
    parser.add_argument("--n_epoch",
                        default=200,
                        type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="batch size")
    parser.add_argument("--optimizer",
                        default="SGD",
                        type=str,
                        help="which optimizer")
    parser.add_argument("--lr", default=0.05, type=float, help="initial lr")
    parser.add_argument("--decay",
                        default=5e-4,
                        type=float,
                        help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--scheduler",
                        default="ReduceLROnPlateau",
                        type=str,
                        help="which scheduler")
    parser.add_argument("--lr_step",
                        default=30,
                        type=int,
                        help="reduce LR per ? epochs")
    parser.add_argument("--lr_lambda",
                        default=0.95,
                        type=float,
                        help="lambda of LambdaLR scheduler")
    parser.add_argument("--lr_factor",
                        default=0.5,
                        type=float,
                        help="factor of lr reduction")
    parser.add_argument("--disable_early_stopping",
                        dest="early_stopping",
                        action="store_false")
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="patience for early stopping")
    parser.add_argument(
        "--tune_on_val",
        default=0.02,
        type=float,
        help=
        "fraction of validation data. If set to 0, use test data as the val data",
    )
    parser.add_argument("--log_auc", dest="log_auc", action="store_true")
    parser.add_argument("--logname",
                        default="vanilla",
                        type=str,
                        help="log name")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--ckpt",
                        default=None,
                        type=str,
                        help="directory for ckpt")
    parser.add_argument(
        "--freeze_extractor",
        dest="freeze_extractor",
        action="store_true",
        help="Whether only training the fc layer",
    )

    # Augmentation
    parser.add_argument(
        "--dis_aug_crop",
        dest="aug_crop",
        action="store_false",
        help="Whether to apply random cropping",
    )
    parser.add_argument(
        "--dis_aug_hflip",
        dest="aug_hflip",
        action="store_false",
        help="Whether to apply horizontally flipping",
    )
    parser.add_argument(
        "--aug_affine",
        dest="aug_affine",
        action="store_true",
        help="Enable random affine",
    )
    parser.add_argument(
        "--aug_rotation",
        type=float,
        default=0,
        help="Maximum degree of the random rotation augmentatiom",
    )
    parser.add_argument("--aug_colorjitter",
                        nargs="*",
                        help="brightness, contrast, saturation, hue")

    # Mixup or InstaHide
    parser.add_argument("--defense_mixup",
                        dest="defense_mixup",
                        action="store_true")
    parser.add_argument("--defense_instahide",
                        dest="defense_instahide",
                        action="store_true")
    parser.add_argument("--klam",
                        default=4,
                        type=int,
                        help="How many images to mix with")
    parser.add_argument("--c_1",
                        default=0,
                        type=float,
                        help="Lower bound of mixing coefs")
    parser.add_argument("--c_2",
                        default=1,
                        type=float,
                        help="Upper bound of mixing coefs")
    parser.add_argument("--use_csprng", dest="use_csprng", action="store_true")
    # GradPrune
    parser.add_argument("--defense_gradprune",
                        dest="defense_gradprune",
                        action="store_true")
    parser.add_argument("--p", default=0.9, type=float, help="prune ratio")
    # Kurtosis
    parser.add_argument("--defense_kurtosis",
                        dest="defense_kurtosis",
                        action="store_true")
    parser.add_argument("--kt_target", default=1.8, type=float, help="Kurtosis target distribution")
    parser.add_argument("--kt_ratio", default=0.01, type=float, help="Kurtosis regularization coefficent")
    parser.add_argument("--kureStepScale",
                        dest="kureStepScale",
                        action="store_true")

    #Apply Noise to Weights
    parser.add_argument("--defense_NoiseWeights",
                        dest="defense_NoiseWeights",
                        action="store_true")
    #We use below parameters max_grad_norm and noise_multiplier

    # DPSGD
    parser.add_argument("--defense_DPSGD",
                        dest="defense_DPSGD",
                        action="store_true")
    parser.add_argument(
        "--delta_list",
        nargs="*",
        default=[1e-3, 1e-4, 1e-5],
        type=float,
        help="Failure prob of DP",
    )
    parser.add_argument("--max_epsilon",
                        default=2,
                        type=float,
                        help="Privacy budget")
    parser.add_argument(
        "--max_grad_norm",
        default=1,
        type=float,
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument("--noise_multiplier",
                        default=1,
                        type=float,
                        help="Noise multiplier")

    parser.add_argument(
        "--n_accumulation_steps",
        default=1,
        type=int,
        help="Run optimization per ? step",
    )
    parser.add_argument(
        "--secure_rng",
        dest="secure_rng",
        action="store_true",
        help=
        "Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    # For attack
    parser.add_argument("--reconstruct_labels", action="store_true")
    parser.add_argument("--attack_lr",
                        default=0.1,
                        type=float,
                        help="learning rate for attack")
    parser.add_argument("--tv", default=0.1, type=float, help="coef. for tv")
    parser.add_argument("--mini",
                        action="store_true",
                        help="use the mini set for attack")
    parser.add_argument("--large",
                        action="store_true",
                        help="use the large set for attack")
    parser.add_argument("--data_seed",
                        default=None,
                        type=int,
                        help="seed to select attack subset")
    parser.add_argument("--attack_epoch",
                        default=0,
                        type=int,
                        help="iterations for the attack")
    parser.add_argument(
        "--bn_reg",
        default=0,
        type=float,
        help="coef. for batchnorm regularization term",
    )
    parser.add_argument(
        "--attacker_eval_mode",
        action="store_true",
        help="use eval model for gradients calculation for attack",
    )
    parser.add_argument(
        "--defender_eval_mode",
        action="store_true",
        help="use eval model for gradients calculation for training",
    )
    parser.add_argument(
        "--BN_exact",
        action="store_true",
        help="use training batch's mean and var",
    )

    args = parser.parse_args()

    hparams = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.decay,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "lr_scheduler": args.scheduler,
        "tune_on_val": args.tune_on_val,
        "batch_size": args.batch_size,
    }

    if args.scheduler == "StepLR":
        hparams["lr_step"] = args.lr_step
        hparams["lr_factor"] = args.lr_factor
    elif args.scheduler == "MultiStepLR":
        hparams["lr_step"] = [100, 150]
        hparams["lr_factor"] = args.lr_factor
    elif args.scheduler == "LambdaLR":
        hparams["lr_lambda"] = args.lr_lambda
    elif args.scheduler == "ReduceLROnPlateau":
        hparams["lr_factor"] = args.lr_factor

    attack_hparams = {
        # Bool settings
        "reconstruct_labels": args.reconstruct_labels,
        "signed_image": args.defense_instahide,
        "mini": args.mini,
        "large": args.large,

        # BN settings
        "BN_exact": args.BN_exact,
        "attacker_eval_mode": args.attacker_eval_mode,
        "defender_eval_mode": args.defender_eval_mode,

        # Hyper-params
        "total_variation": args.tv,
        "epoch": args.attack_epoch,
        "bn_reg": args.bn_reg,
        "attack_lr": args.attack_lr,

        "noise_multiplier": args.noise_multiplier,
    }

    return args, hparams, attack_hparams


def parse_augmentation(args):
    return {
        "hflip":
        args.aug_hflip,
        "crop":
        args.aug_crop,
        "rotation":
        args.aug_rotation,
        "color_jitter": [float(i) for i in args.aug_colorjitter]
        if args.aug_colorjitter is not None else None,
        "affine":
        args.aug_affine,
    }


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def save_fig(img_arr, fname, save_npy=False, save_fig=True):
    if torch.is_tensor(img_arr):
        img_arr = img_arr.cpu().detach()
    if len(img_arr.shape) == 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))
    elif len(img_arr.shape) == 4:
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
    # print(img_arr.shape)
    if save_npy:
        np.save(re.sub(r"(jpg|png|pdf)\b", "npy", fname), img_arr)
    if save_fig:
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        plt.imshow(img_arr)
        plt.axis("off")
        plt.savefig(fname, bbox_inches="tight")
        plt.show()


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    # view = [1] * len(x.shape)
    # view[1] = -1
    # print(view)
    x = (x - bn_mean) / torch.sqrt(bn_var + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var != 0).float()
    return x


class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats=None, n_features=1024):
        super(StandardizeLayer, self).__init__()
        if bn_stats is None:
            mean = np.zeros(n_features, dtype=np.float32)
            var = np.ones(n_features, dtype=np.float32)
            bn_stats = (torch.from_numpy(mean), torch.from_numpy(var))
        self.bn_stats = bn_stats

    def forward(self, x):
        self.bn_stats = (self.bn_stats[0].to(x.device),
                         self.bn_stats[1].to(x.device))
        # print(x.shape)
        return standardize(x, self.bn_stats)


def cross_entropy_for_onehot(pred, target):
    # Prediction should be logits instead of probs
    return torch.mean(torch.sum(-target * log_softmax(pred, dim=-1), 1))


def patch_image(x, dim=(32, 32)):
    """Patch a batch of images for better visualization, keeping images in rows of 8. If the number of images isn't divisible by 8, pads the remaining space with whitespace."""
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()
    batch_size = len(x)
    if batch_size == 1:
        return x[0]
    else:
        if batch_size % 8 != 0:
            # Pad batch with white squares
            pad_size = int(math.ceil(batch_size / 8) * 8) - batch_size
            x = np.append(x, np.zeros((pad_size, *x[0].shape)), axis=0)
        batch_size = len(x)
        x = np.transpose(x, (0, 2, 3, 1))
        if int(np.sqrt(batch_size))**2 == batch_size:
            s = int(np.sqrt(batch_size))
            x = np.reshape(x, (s, s, dim[0], dim[1], 3))
            x = np.concatenate(x, axis=2)
            x = np.concatenate(x, axis=0)
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.reshape(x, (8, batch_size // 8, dim[0], dim[1], 3))
            x = np.concatenate(x, axis=2)
            x = np.concatenate(x, axis=0)
            x = np.transpose(x, (2, 0, 1))
    return x

# class KurtosisWeight:
#     def __init__(self, weight_tensor, name, kurtosis_target=1.9, k_mode='avg'):
#         self.kurtosis_loss = 0
#         self.kurtosis = 0
#         self.weight_tensor = weight_tensor
#         self.name = name
#         self.k_mode = k_mode
#         self.kurtosis_target = kurtosis_target

#     def fn_regularization(self):
#         return self.kurtosis_calc()

#     def kurtosis_calc(self):
#         #print("--------------------------------------------------------------------------------")
#         #torch.set_printoptions(profile="default")

#         self.weight_tensor = torch.flatten(self.weight_tensor)

#         #print("Layer name: ", self.name)
#         #print("weight_tensor: ", torch.flatten(self.weight_tensor))
#         mean_output = torch.mean(self.weight_tensor)
#         #print("Mean output:", mean_output)
#         std_output = torch.std(self.weight_tensor)
#         #print("STD output:", std_output)
#         kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
#         tempVar = torch.flatten((self.weight_tensor - mean_output)/ std_output)
#         #print("Weight tensor - mean / std : ", tempVar)
#         #print("max(Weight tensor - mean / std ): ", torch.max(tempVar))
#         # lowerBound = int(torch.numel(tempVar)/2 - 5)
#         # upperBound = lowerBound + 10
#         # print("Middle 10 elements of max(Weight tensor - mean / std ): ", tempVar[lowerBound:upperBound])
#         #print("(Weight tensor - mean / std ) **4: ", tempVar **4)
#         #print("max(Weight tensor - mean / std ) **4)", torch.max(tempVar**4))
#         #print("mean((Weight tensor - mean / std ) **4):", kurtosis_val)
#         self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
#         #print("Kurtosis loss: ", self.kurtosis_loss)
#         self.kurtosis = kurtosis_val

#         if self.k_mode == 'avg':
#             self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
#             #print("Averaged kurtosis loss: ", self.kurtosis_loss)
#             #print("--------------------------------------------------------------------------------")
#             self.kurtosis = torch.mean(kurtosis_val)
#         elif self.k_mode == 'max':
#             self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
#             self.kurtosis = torch.max(kurtosis_val)
#         elif self.k_mode == 'sum':
#             self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
#             self.kurtosis = torch.sum(kurtosis_val)

# def find_weight_tensor_by_name(model, name_in):
#     for name, param in model.named_parameters():
#         # print("name_in: " + str(name_in) + " name: " + str(name))
#         if name == name_in:
#             return param

# def kurtosis(modelparams,kt_target=1.8, kt_ratio=0.01):
#         regLoss = 0
#         count = 0
#         for name,wtensor in modelparams:
#             #wtensor = torch.flatten(weight_tensor)
#             #print(name)
#             mean = torch.mean(wtensor)
#             std = torch.std(wtensor)
#             kurt = torch.mean((( wtensor - mean ) / std) ** 4 )
#             if torch.isnan(kurt):
#                 regLoss += 0
#                 print("Rejected:", name)
#             elif "bias" in name :
#                 #print("Accepted:", name)
#                 regLoss += torch.mean((kurt - kt_target)**2)
#                 count += 1
#             else:
#                 regLoss += 0

#         if count == 0:
#             return 0
#         else:
#             regLoss =regLoss / count
#         #regLoss = regLoss / 19
#         #print(regLoss)
        
#         return kt_ratio * regLoss

# def cross_entropy_with_kurtosis(wrappedModule, input,target, kt_target, kt_ratio):
#     model = wrappedModule._model  
#     weight_to_hook = {}
#     all_convs = [n.replace(".wrapped_module", "") + '.weight' for n, m in model.named_modules() if isinstance(m, nn.Linear)]# or isinstance(m, nn.Conv2d)]
#     weight_name = all_convs#[1:] #[1:]

#     #print(weight_name)
#     for name in weight_name:
#         # pdb.set_trace()
#         curr_param = find_weight_tensor_by_name(model, name)
#         # if not curr_param:
#         #     name = 'float_' + name # QAT name
#         #     curr_param = fine_weight_tensor_by_name(self.model, name)
#         # if curr_param is not None:
#         weight_to_hook[name] = curr_param
#         #print(name)

#     hookF_weights = {}
#     #print(weight_to_hook.items())
#     for name, w_tensor in weight_to_hook.items():
#         # pdb.set_trace()
#         hookF_weights[name] = KurtosisWeight(w_tensor, name, kurtosis_target=kt_target,
#                                                 k_mode='avg')

#     w_kurtosis_regularization = 0
#     w_temp_values = []
#     w_kurtosis_loss = 0
#     count = 0 
#     for w_kurt_inst in hookF_weights.values():
#         count = count +1
#         w_kurt_inst.fn_regularization()
#         w_temp_values.append(w_kurt_inst.kurtosis_loss)
    
#     ClassificationLoss = cross_entropy(input,target, reduction = 'mean')
#     if (w_temp_values == []):
#         return ClassificationLoss
#     w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
#     w_kurtosis_loss = w_kurtosis_loss / count #For ResNet18 SHOULD BE 19 BUT 20 when ADDING nn.Linear, ADJUST If doing all Layers
#     #print(w_kurtosis_loss)
#     w_kurtosis_regularization = kt_ratio * w_kurtosis_loss
#     #print("KurtosisLoss: ",w_kurtosis_regularization)
#     #ClassificationLoss = cross_entropy(input,target, reduction = 'mean')
#     #KurtosisLoss = kurtosis(model.named_parameters(),kt_target,kt_ratio)
#     wrappedModule.log(
#             "step/KURE_unscaled",
#             w_kurtosis_loss,
#             on_step=True,
#             on_epoch=False,
#             prog_bar=True,
#             logger=True,
#         )
    
#     wrappedModule.log(
#             "step/KURE_scaled",
#             w_kurtosis_regularization,
#             on_step=True,
#             on_epoch=False,
#             prog_bar=True,
#             logger=True,
#         )

#     return ClassificationLoss + w_kurtosis_regularization #KurtosisLoss



# def cross_entropy_kurtosis_grads(wrappedModule, input,target, kt_target, kt_ratio):
#     model = wrappedModule._model    
#     model.zero_grad()
#     ClassificationLoss = cross_entropy(input,target, reduction = 'mean')

    
#     #ClassificationGradients = torch.autograd.grad(ClassificationLoss, input, retain_graph=True, create_graph=True)[0]
#     ClassificationLoss.backward(retain_graph=True)


#     weight_to_hook = {}
#     all_convs = [n.replace(".wrapped_module", "") + '.weight' for n, m in model.named_modules() if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear)]
#     weight_name = all_convs[1:]
#     for name in weight_name:
#         curr_param = find_weight_tensor_by_name(model, name)
#         weight_to_hook[name] = curr_param.grad
#         weight_to_hook[name].requires_grad_()

#     hookF_weights = {}
#     for name, g_tensor in weight_to_hook.items():
#         hookF_weights[name] = KurtosisWeight(g_tensor, name, kurtosis_target=kt_target,
#                                                 k_mode='avg')

#     w_kurtosis_regularization = 0
#     w_temp_values = []
#     w_kurtosis_loss = 0
#     count = 0
#     for w_kurt_inst in hookF_weights.values():
#         count = count+1
#         w_kurt_inst.fn_regularization()
#         w_temp_values.append(w_kurt_inst.kurtosis_loss)

#     w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
#     w_kurtosis_loss = w_kurtosis_loss / count #For ResNet18 SHOULD BE 19 BUT 20 when ADDING nn.Linear, ADJUST If doing all Layers
#     #print("Kurt Loss:",w_kurtosis_loss)
#     w_kurtosis_regularization = kt_ratio * w_kurtosis_loss

#     wrappedModule.log(
#             "step/KURE_unscaled",
#             w_kurtosis_loss,
#             on_step=True,
#             on_epoch=False,
#             prog_bar=True,
#             logger=True,
#         )
    
#     wrappedModule.log(
#             "step/KURE_scaled",
#             w_kurtosis_regularization,
#             on_step=True,
#             on_epoch=False,
#             prog_bar=True,
#             logger=True,
#         )


#     return ClassificationLoss + w_kurtosis_regularization


def find_tensor_by_name(model, name_in):
    for name, param in model.named_parameters():
        # print("name_in: " + str(name_in) + " name: " + str(name))
        if name == name_in:
            return param

def checkGradients(model, arg):
    weight_to_hook = {}
    all_convs = [n.replace(".wrapped_module", "") + '.weight' for n, m in model.named_modules() if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear)]
    weight_name = all_convs[1:]
    for name in weight_name:
        curr_param = find_tensor_by_name(model, name)
        weight_to_hook[name] = curr_param.grad
        weight_to_hook[name].requires_grad_()
        ##PRINT SIGNS OF GRADIENTS
        #print(arg, " Loss Gradient for ",name," :",torch.sign(curr_param.grad))
    return weight_to_hook

def compareGradients(grad1,grad2):
    for  g_tensor, g_tensor2 in zip(grad1.items(),grad2.items()):
        if (g_tensor[0] == g_tensor2[0]):
            print(g_tensor[0], " :", torch.ne(torch.sign(g_tensor[1]),torch.sign(g_tensor2[1])))
            print("Difference: ----------------------------")
            print(torch.abs(g_tensor[1] - g_tensor2[1]))

def compareGradientsJacob(wrappedModule, grad1,grad2):
    grad1x = torch.cat([torch.flatten(g_tensor) for g_tensor in grad1])
    grad2x = torch.cat([torch.flatten(g_tensor) for g_tensor in grad2])
    signMatrix = torch.ne(torch.sign(grad1x),torch.sign(grad2x))
    #print("Gradient values with flipped signs (True if changed): ---------")
    #print(signMatrix)
    #print ("Number of flipped gradient value signs:", torch.numel(signMatrix[signMatrix==True]))
    #print("Percentage of flipped gradient signs:", (torch.numel(signMatrix[signMatrix==True]) / torch.numel(signMatrix)) * 100, "%" )
    #print("Gradient Value Difference: ----------------------------")
    #print(torch.abs(grad1 - grad2))

    wrappedModule.log(
        "step/numFlippedSigns",
        torch.numel(signMatrix[signMatrix==True]),
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        logger=True,
    )
    
    wrappedModule.log(
       "step/PercentageFlippedSigns",
        (torch.numel(signMatrix[signMatrix==True]) / torch.numel(signMatrix)) * 100,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        logger=True,
    )

    