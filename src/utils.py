import torch
import torchvision

# ## Fist version with device argument (as in code provided by the authors)
# def load_pretrained_model(path, device='cuda'):
#     model = torchvision.models.__dict__['resnet18'](pretrained=False)

#     state = torch.load(path, map_location=device)

#     state_dict = state['state_dict']
#     for key in list(state_dict.keys()):
#         state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
  
#     model_dict = model.state_dict()
#     weights = {k: v for k, v in state_dict.items() if k in model_dict}
#     if weights == {}:
#         print('No weight could be loaded..')
#     model_dict.update(weights)
#     model.load_state_dict(model_dict)

#     model.fc = torch.nn.Sequential()

#     return model

## Second version with no device specification. 
def load_pretrained_model(path):
    model = torchvision.models.__dict__['resnet18'](pretrained=False)

    state = torch.load(path)

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
  
    model_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    model.fc = torch.nn.Sequential()

    return model


