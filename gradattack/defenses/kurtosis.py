import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from functools import reduce

class KurtosisWeight:
    def __init__(self, weight_tensor, name, kurtosis_target=1.9, k_mode='avg'):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.name = name
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

    def fn_regularization(self):
        return self.kurtosis_calc()

    def kurtosis_calc(self):
        #print("--------------------------------------------------------------------------------")
        #torch.set_printoptions(profile="default")

        self.weight_tensor = torch.flatten(self.weight_tensor)

        #print("Layer name: ", self.name)
        #print("weight_tensor: ", torch.flatten(self.weight_tensor))
        mean_output = torch.mean(self.weight_tensor)
        #print("Mean output:", mean_output)
        std_output = torch.std(self.weight_tensor)
        #print("STD output:", std_output)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        #tempVar = torch.flatten((self.weight_tensor - mean_output)/ std_output)
        #print("Weight tensor - mean / std : ", tempVar)
        #print("max(Weight tensor - mean / std ): ", torch.max(tempVar))
        # lowerBound = int(torch.numel(tempVar)/2 - 5)
        # upperBound = lowerBound + 10
        # print("Middle 10 elements of max(Weight tensor - mean / std ): ", tempVar[lowerBound:upperBound])
        #print("(Weight tensor - mean / std ) **4: ", tempVar **4)
        #print("max(Weight tensor - mean / std ) **4)", torch.max(tempVar**4))
        #print("mean((Weight tensor - mean / std ) **4):", kurtosis_val)
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        #print("Kurtosis loss: ", self.kurtosis_loss)
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            #print("Averaged kurtosis loss: ", self.kurtosis_loss)
            #print("--------------------------------------------------------------------------------")
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)

def find_weight_tensor_by_name(model, name_in):
    for name, param in model.named_parameters():
        # print("name_in: " + str(name_in) + " name: " + str(name))
        if name == name_in:
            return param

def kurtosis(modelparams,kt_target=1.8, kt_ratio=0.01):
        regLoss = 0
        count = 0
        for name,wtensor in modelparams:
            #wtensor = torch.flatten(weight_tensor)
            #print(name)
            mean = torch.mean(wtensor)
            std = torch.std(wtensor)
            kurt = torch.mean((( wtensor - mean ) / std) ** 4 )
            if torch.isnan(kurt):
                regLoss += 0
                print("Rejected:", name)
            elif "bias" in name :
                #print("Accepted:", name)
                regLoss += torch.mean((kurt - kt_target)**2)
                count += 1
            else:
                regLoss += 0

        if count == 0:
            return 0
        else:
            regLoss =regLoss / count
        #regLoss = regLoss / 19
        #print(regLoss)
        
        return kt_ratio * regLoss

def cross_entropy_with_kurtosis(wrappedModule, input,target, kt_target, kt_ratio):
    model = wrappedModule._model  
    weight_to_hook = {}
    all_convs = [n.replace(".wrapped_module", "") + '.weight' for n, m in model.named_modules() if isinstance(m, nn.Linear)]# or isinstance(m, nn.Conv2d)]
    weight_name = all_convs#[1:] #[1:]

    #print(weight_name)
    for name in weight_name:
        # pdb.set_trace()
        curr_param = find_weight_tensor_by_name(model, name)
        # if not curr_param:
        #     name = 'float_' + name # QAT name
        #     curr_param = fine_weight_tensor_by_name(self.model, name)
        # if curr_param is not None:
        weight_to_hook[name] = curr_param
        #print(name)

    hookF_weights = {}
    #print(weight_to_hook.items())
    for name, w_tensor in weight_to_hook.items():
        # pdb.set_trace()
        hookF_weights[name] = KurtosisWeight(w_tensor, name, kurtosis_target=kt_target,
                                                k_mode='avg')

    w_kurtosis_regularization = 0
    w_temp_values = []
    w_kurtosis_loss = 0
    count = 0 
    for w_kurt_inst in hookF_weights.values():
        count = count +1
        w_kurt_inst.fn_regularization()
        w_temp_values.append(w_kurt_inst.kurtosis_loss)
    
    ClassificationLoss = cross_entropy(input,target, reduction = 'mean')
    if (w_temp_values == []):
        return ClassificationLoss
    w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
    w_kurtosis_loss = w_kurtosis_loss / count #For ResNet18 SHOULD BE 19 BUT 20 when ADDING nn.Linear, ADJUST If doing all Layers
    #print(w_kurtosis_loss)
    w_kurtosis_regularization = kt_ratio * w_kurtosis_loss
    #print("KurtosisLoss: ",w_kurtosis_regularization)
    #ClassificationLoss = cross_entropy(input,target, reduction = 'mean')
    #KurtosisLoss = kurtosis(model.named_parameters(),kt_target,kt_ratio)
    wrappedModule.log(
            "step/KURE_unscaled",
            w_kurtosis_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
    
    wrappedModule.log(
            "step/KURE_scaled",
            w_kurtosis_regularization,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    return ClassificationLoss + w_kurtosis_regularization #KurtosisLoss



def cross_entropy_kurtosis_grads(wrappedModule, input,target, kt_target, kt_ratio):
    #model = wrappedModule._model    
    #model.zero_grad()
    input = input.requires_grad_(True)
    ClassificationLoss = cross_entropy(input,target, reduction = 'mean')

    input_grad, = torch.autograd.grad(ClassificationLoss, input, retain_graph=True, create_graph = True)

    #wrappedModule.manual_backward(ClassificationLoss, retain_graph=True)
    count = 0
    hookF_weights = {}
    weight_to_hook = {}
    for  g_tensor in input_grad:
        count = count + 1
        name=str(count)
        hookF_weights[name] = KurtosisWeight(g_tensor, name, kurtosis_target=kt_target,
                                                k_mode='avg')
        #print(g_tensor)
        weight_to_hook[name] = g_tensor
    wrappedModule.tempClassGrads = input_grad.detach().clone()


    # weight_to_hook = {}
    # all_convs = [n.replace(".wrapped_module", "") + '.weight' for n, m in model.named_modules() if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear)]
    # weight_name = all_convs[1:]
    # for name in weight_name:
    #     curr_param = find_weight_tensor_by_name(model, name)
    #     weight_to_hook[name] = curr_param.grad
    #     weight_to_hook[name].requires_grad_()
    #     #print(curr_param.grad)
    #wrappedModule.tempClassGrads = weight_to_hook

    # hookF_weights = {}
    # for name, g_tensor in weight_to_hook.items():
    #     hookF_weights[name] = KurtosisWeight(g_tensor, name, kurtosis_target=kt_target,
    #                                              k_mode='avg')

    w_kurtosis_regularization = 0
    w_temp_values = []
    w_kurtosis_loss = 0
    count = 0
    for w_kurt_inst in hookF_weights.values():
        count = count+1
        w_kurt_inst.fn_regularization()
        w_temp_values.append(w_kurt_inst.kurtosis_loss)

    w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
    w_kurtosis_loss = w_kurtosis_loss / count #For ResNet18 SHOULD BE 19 BUT 20 when ADDING nn.Linear, ADJUST If doing all Layers
    #print("Kurt Loss:",w_kurtosis_loss)
    w_kurtosis_regularization = kt_ratio * w_kurtosis_loss

    wrappedModule.log(
            "step/KURE_unscaled",
            w_kurtosis_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
    
    wrappedModule.log(
            "step/KURE_scaled",
            w_kurtosis_regularization,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )


    return ClassificationLoss + w_kurtosis_regularization