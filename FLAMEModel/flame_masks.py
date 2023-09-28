import pickle
import os

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get_flame_mask():
    """
    Different mask options
    : boundary, eye region, face, forehead, left ear, left eye region, left eyeball,
    lips, neck, nose, right ear, nose, right ear, right eye region, right eye ball, scalp

    """
    mask_path = "FLAMEModel/model/FLAME_masks.pkl"
    with open(mask_path, 'rb') as f:
        ss = pickle.load(f, encoding='latin1')
        masks = Struct(**ss)
    return masks