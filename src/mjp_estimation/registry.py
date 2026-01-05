from typing import Union
from pathlib import Path
from .models import *
from .jackson_model import *

MODEL_REGISTRY = {
    "SingleServerQueue": SingleServerQueue,
    "SingleServerQueueOneParameter": SingleServerQueueOneParameter,
    "TandemQueuePeriodicArrivalsConstantService": TandemQueuePeriodicArrivalsConstantService,
    "TandemQueuePeriodicArrivalsConstantFixedService": TandemQueuePeriodicArrivalsConstantFixedService,
    "TandemQueuePeriodicArrivalsConstantFixedService": TandemQueuePeriodicArrivalsConstantFixedService,
    "TandemQueueSecularTrendArrivalsConstantFixedService": TandemQueueSecularTrendArrivalsConstantFixedService,
    "TandemQueueSecularTrendArrivalsFixedService": TandemQueueSecularTrendArrivalsFixedService
}

def load_model(filepath: Union[str, Path]):
    """
    Loads a model file, identifies the correct class,
    and instantiates it with the saved parameters.
    """
    # print(f"\n--- Loading model from {filepath} ---")
    try:
        with np.load(filepath) as data:
            # 1. Load the class name string
            # .item() converts 0-d array back to string
            class_name = data['class_name'].item()
            
            # 2. Look up the class object in our registry
            model_class: MJPModel = MODEL_REGISTRY.get(class_name)
            
            if model_class is None:
                raise ValueError(
                    f"Unknown class name '{class_name}' in file. "
                    f"Is it in MODEL_REGISTRY?"
                )
        # 3. Dynamically instantiate the correct class
        return model_class.load(filepath)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except KeyError as e:
        print(f"Error: File {filepath} is corrupt or missing key: {e}")
        raise