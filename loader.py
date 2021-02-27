"""
simple retina model loader


"""
from collections import OrderedDict
import os
import os.path as osp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace


# pylint: disable=no-member
# pylint: disable=not-callable
class Retina:
    def __init__(self, model="resnet50", head_mode=7, loss_mode=7, **kwargs):
        """
            loss_mode   ignores loss from specific heads if < 7
            head_mode   computes some heads without gradient if  < 7
            config and args ought to be together!!! not separate collections of hyperparameters
        """
        self.cfg = self.get_config(model, **kwargs)
        self.args = self.get_args(**kwargs)


        self.net = RetinaFace(cfg=self.cfg, head_mode=head_mode)
        self.load_checkpoint()
        self._state_dict = self.detach_state_dict()
        self.set_enviro()

        self.loss_fn = MultiBoxLoss(self.args["num_classes"], 0.35, True, 0, True, 7, 0.35, False,
                                    head_mode=head_mode)
        self.optim = optim.SGD(self.net.parameters(), lr=self.args["lr"],
                               momentum=self.args["momentum"],
                               weight_decay=self.args["weight_decay"])

        self.priors = self.get_priors()
        self.dataset = self.get_dataset()

        self.loss_mode = loss_mode

        # initialied in train step, train epoch, or ad hoc
        self.loader = None
        self.images = None
        self.targets = None
        # self.loss_weights = torch.tensor([self.cfg['loc_weight'], 1., 1.])

    #def memir() -> memir.py

    def detach_state_dict(self):
        dic = self.net.state_dict()
        return {key:dic[key].cpu().clone().detach() for key in dic}

    def train_step(self, **kwargs):

        loss_mode = self.loss_mode if "loss_mode" not in kwargs else kwargs["loss_mode"]
        images, targets, out = self.fwd(**kwargs)
        loss = self.err(out, targets, loss_mode)
        self.bkw(loss)

    def reset_optim(self, **kwargs):
        _kw = {"lr": self.args["lr"], "momentum":self.args["momentum"],
               "weight_decay":self.args["weight_decay"]}
        __kw = {k:kwargs[k] for k in kwargs if k in _kw and kwargs[k] != _kw[k]}
        _kw.update(__kw)
        self.optim = optim.SGD(self.net.parameters(), **_kw)

    def load_batch(self, **kwargs):
        _kw = {"lr": self.args["lr"], "momentum":self.args["momentum"],
               "weight_decay":self.args["weight_decay"]}
        __kw = {k:kwargs[k] for k in kwargs if k in _kw and kwargs[k] != _kw[k]}
        if __kw:
            _kw.update(__kw)
            self.optim = optim.SGD(self.net.parameters(), **_kw)
        if self.loader is None:
            self.loader = self.get_loader(**kwargs)

        images, targets = next(self.loader)
        if self.cfg["gpu_train"]:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        return images, targets

    def fwd(self, **kwargs):
        images, targets = self.load_batch(**kwargs)
        return images, targets, self.net(images)

    def bkw(self, loss):
        assert self.args["mode"] == "train", "set mode to train >>> self.update_args(mode='train')"
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def err(self, out, targets, loss_mode=7):
        """ loss_mode, bitmask
            7   loss_b  loss_c  loss_landm
            6   0       loss_c  loss_landm
            5   loss_b  0       loss_landm
            4   0       0       loss_landm
            3   loss_b  loss_c  0
            2   0       loss_c  0
            1   loss_b  0       0
        """
        if loss_mode < 1 or loss_mode > 7:
            loss_mode = 7
        _loss = self.loss_fn(out, self.priors, targets)
        _losses = {1:_loss[0]*self.cfg['loc_weight'],
                   2:_loss[1],
                   4:_loss[2]}
        return sum([_losses[k] for k in [k for k in _losses if loss_mode & k]])
        # loss_l, loss_c, loss_landm = self.loss_fn(out, self.priors, targets)
        # return self.cfg['loc_weight'] * loss_l + loss_c + loss_landm

    def get_loader(self, **kwargs):
        _kw = {"batch_size":self.cfg["batch_size"], "num_workers": self.args["num_workers"],
               "shuffle":True}
        __kw = {k:kwargs[k] for k in kwargs if k in _kw and kwargs[k] != _kw[k]}
        _kw.update(__kw)
        return iter(data.DataLoader(self.dataset, collate_fn=detection_collate, **_kw))

    def update_args(self, **kwargs):
        self.images = None
        self.targets = None
        self.loader = None
        _kw = {k:kwargs[k] for k in kwargs if k in self.cfg}
        self.cfg.update(_kw)
        _kw = {k:kwargs[k] for k in kwargs if k in self.args}
        self.args.update(_kw)

    def get_dataset(self):
        return WiderFaceDetection(self.args['txt_path'], preproc(self.cfg['image_size'],
                                                                 self.args['rgb_mean']),
                                  relative_path=self.args['relative_path'])

    def get_priors(self):
        """ hm, spaghetti
            # cfg['min_sizes']
            # self.steps = cfg['steps']
            # self.clip = cfg['clip']
            # cfg['image_size']
        """
        priorbox = PriorBox(self.cfg, image_size=(self.cfg['image_size'], self.cfg['image_size']))
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.cuda()
        return priors

    def get_config(self, model="resnet50", **kwargs):
        """
            cfg from retinaface, override with kwargs option
            kwargs in:
                'name': 'Resnet50',
                'min_sizes': [[16, 32], [64, 128], [256, 512]],
                'steps': [8, 16, 32],
                'variance': [0.1, 0.2],
                'clip': False,
                'loc_weight': 2.0,
                'gpu_train': True,
                'batch_size': 24,
                'ngpu': 4,
                'epoch': 100,
                'decay1': 70,
                'decay2': 90,
                'image_size': 840,
                'pretrain': True,
                'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
                'in_channel': 256,
                'out_channel': 256
        """
        assert model in ("resnet50", "mobile0.25"), "no model config: try resnet50 or mobile0.25"
        cfg = {"resnet50":cfg_re50, "mobile0.25":cfg_mnet}[model]
        _kw = {k:kwargs[k] for k in kwargs if k in cfg}
        cfg.update(_kw)
        return cfg

    def get_args(self, **kwargs):
        """  should be with cfg, but to avoid poor loading, dont combine
        """
        args = {"txt_path":
                "/media/z/Elements/data/Face/WIDER/widerface_retinaface/train/label.txt",
                "relative_path":"../../WIDER_train/images/",
                "save_folder":"/home/z/share/RetinaFace",
                "rgb_mean":(104, 117, 123),
                "num_workers":4,
                "lr":1e-3,
                "momentum":0.9,
                "weight_decay":5e-4,
                "gamma":0.1,
                "num_classes":2,
                "checkpoint":None,
                "epoch":0,
                "mode":"train"}
        _kw = {k:kwargs[k] for k in kwargs if k in args}
        args.update(_kw)

        # paths
        assert osp.isfile(args["txt_path"]), "train dataset labels not found <%s>"%args["txt_path"]
        _images = osp.abspath(args["txt_path"].replace("label.txt", args["relative_path"]))
        assert osp.isdir(_images), "train dataset images not found <%s>"%_images

        if not osp.isdir(args["save_folder"]):
            os.makedirs(args["save_folder"])
        return args

    def load_checkpoint(self):
        """ checkpoint loader, if checkpoint exists
        """
        checkpoint = self.args["checkpoint"]
        if checkpoint is not None and osp.isfile(checkpoint):
            state_dict = torch.load(checkpoint)

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.net.load_state_dict(new_state_dict)
            print(" RetinaNet() loaded checkpoint", checkpoint)

    def set_enviro(self):
        if self.cfg["gpu_train"]:
            num_gpu = min(self.cfg["ngpu"], torch.cuda.device_count())
            if num_gpu > 1:
                self.net = torch.nn.DataParallel(self.net).cuda()
            else:
                self.net = self.net.cuda()
            cudnn.benchmark = True

        if self.args["mode"] == "train":
            self.net.train()
        else:
            self.net.eval()
