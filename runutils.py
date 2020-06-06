"""
runtime code repeated in various tests
"""
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
try:
    import accimage # its a bit faster
    _backend = "accimg"
except:
    pass

def open_accimg(name, debug=False):
    out=None
    if 'accimage' in sys.modules:
        _im = accimage.Image(name)
        out = np.zeros([_im.channels, _im.height, _im.width], dtype=np.uint8)
        _im.copyto(out)
        if debug:
            print("using accimg")
    return out

def open_pilimg(name, debug=False):
    out=None
    if 'PIL.Image'in sys.modules:
        _im = Image.open(name)
        out = np.frombuffer(_im.tobytes(), dtype=np.uint8).reshape(len(_im.getbands()), *_im.size[::-1])
        if debug:
            print("using pil")
    return out

def open_tensor(name, dtype="float32", device="cuda", normed=False, center=None, backend=None, debug=False):
    """
    Args:
        name    (str) valid filename string
        dtype   (str [float32]) in ("uint8", "float16", "float32", "float64")
        normed  (bool [False]) if True out/=255.
        center  (tuple [None]) 
    Open Torch tensor without transposing
    """
    _dtypes = ("uint8", "float16", "float32", "float64")
    assert dtype in _dtypes, "only %s accepted"%str(_dtypes)
    dtype = torch.__dict__[dtype]
    _norm = 1.0

    out = None
    if backend is None or backend == "accimg" and 'accimage' in sys.modules:
        try:
            out = open_accimg(name, debug=debug)
        except:
            print("accimg could not open file")

    if out is None:
        try:
            out = open_pilimg(name, debug=debug)
        except:
            print("pil could not open file")

    out = torch.from_numpy(out).to(device=device).unsqueeze(0).to(dtype=dtype)
    if normed and dtype != torch.uint8:
        out.div_(255.)
        _norm = 255.

    if center is not None:
        _msg = "center needs to be types (tuple, list), found <%s>"%str(type(center))
        assert isinstance(center, (list, tuple)), _msg
        if max(center) <= 1:
            _norm = 1.0 if _norm > 1.0 else 1./255.

        out.sub_(torch.tensor([center], device="cuda", dtype=dtype).view(1, 3, 1, 1).div_(_norm))
    return out

def mean_center(img, center=(104, 117, 123), norm=False):
    """
        mean center from detect / test_widerface
        # curios data is not centered to (-1, 1) or (-0.5, 0.5)
    """
    _scale = 255.0 if norm else 1.0
    img = (img.astype(np.float32) - np.array([[center]], dtype=np.float32))/_scale

    return (torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)).to(dtype=torch.float32).contiguous()

##
#
# from test_widerface or detect
#
def check_keys(model, pretrained_state_dict, verbose=False):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if missing_keys or verbose:
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix, verbose=False):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    if verbose:
        print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, device="cuda", verbose=False):
    """
    device = str, or dict: "cpu", "cuda", "cuda:0", {"cuda:0", "cuda:1"}
    """
    if verbose:
        print('Loading pretrained model from {}'.format(pretrained_path))

    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
