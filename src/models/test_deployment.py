
import torchvision.models as models
import torch
import os

FILE_PATH = os.path.dirname(__file__) 
RAW_MODELS_PATH = os.path.join(FILE_PATH, '../../models/')

def main():
    model = models.resnet101(pretrained=True)
    script_model = torch.jit.script(model)
    script_model.save(os.path.join(RAW_MODELS_PATH, 'deployable_model.pt'))

if __name__ == "__main__":
    main()
