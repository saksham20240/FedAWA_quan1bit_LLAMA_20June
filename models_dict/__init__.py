# models_dict/__init__.py - Updated for Medical Federated Learning

# Import all existing models (unchanged)
from .cnn import (
    CNNCifar100, CNNCifar10, LeNet5, MLP,
    CNNCifar100_fedlaw, CNNCifar10_fedlaw, 
    LeNet5_fedlaw, MLP_fedlaw
)

from .resnet import (
    ResNet18, ResNet20, ResNet56, ResNet110,
    ResNet18_fedlaw, ResNet20_fedlaw, ResNet56_fedlaw, ResNet110_fedlaw,
    WRN56_2, WRN56_4, WRN56_8,
    WRN56_2_fedlaw, WRN56_4_fedlaw, WRN56_8_fedlaw
)

from .densenet import (
    DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar,
    DenseNet121_fedlaw, DenseNet169_fedlaw, DenseNet201_fedlaw, 
    DenseNet161_fedlaw, densenet_cifar_fedlaw
)

from .vit import (
    ViT, ViT_fedlaw
)

# Import new Llama models for medical federated learning
from .llama_models import (
    MedicalLlama7B, MedicalLlama3B,
    MedicalLlama7B_fedlaw, MedicalLlama3B_fedlaw,
    Llama7B_medical, Llama3B_medical,
    Llama7B_medical_fedlaw, Llama3B_medical_fedlaw,
    CNNfmnist_medical, CNNfmnist_fedlaw_medical
)

from .reparam_function import ReparamModule

# Add missing CNNfmnist from cnn.py if not already there
try:
    from .cnn import CNNfmnist, CNNfmnist_fedlaw
except ImportError:
    # If CNNfmnist doesn't exist in cnn.py, use medical versions
    CNNfmnist = CNNfmnist_medical
    CNNfmnist_fedlaw = CNNfmnist_fedlaw_medical
