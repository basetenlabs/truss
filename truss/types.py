from enum import Enum


class ModelFrameworkType(Enum):
    SKLEARN = 'sklearn'
    TENSORFLOW = 'tensorflow'
    KERAS = 'keras'
    PYTORCH = 'pytorch'
    HUGGINGFACE_TRANSFORMER = 'huggingface_transformer'
    CUSTOM = 'custom'
