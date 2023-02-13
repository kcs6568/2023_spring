import argparse

class TrainParser:
    def __init__(self) -> None:
        self.args = self._make_parser()
        
    def _make_parser(self):
        parser = argparse.ArgumentParser()
        
        # task settings
        parser.add_argument("--task", nargs='+', default=[])
        parser.add_argument("--cfg", type=str, default=None)
        parser.add_argument("--general", action='store_true')
        
        ##### seperate training
        parser.add_argument("--seperate", action='store_true')
        parser.add_argument("--seperate-task", default=None)
        
        ##### loss balancing
        parser.add_argument("--lossbal", action='store_true')
        parser.add_argument("--loss-ratio", nargs='+', default=None, type=int)
        
        ##### step by step training
        parser.add_argument("--step-train", action='store_true')
        parser.add_argument("--task-warmup-ratio", nargs='+', default=[0.6, 0.6, 1])
        
        # dataset settings
        parser.add_argument("--task-bs", nargs='+', default=[], type=int)
        parser.add_argument(
            "-b", "--batch-size", default=4, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
        parser.add_argument(
            "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
        )
        parser.add_argument(
            "--use_testset",
            action="store_true",
        )
        parser.add_argument(
            "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
        )
        parser.add_argument("--use-minids", action="store_true", help="Use COCO minitrain dataset")
        parser.add_argument("--pin-memory", action='store_false')
        parser.add_argument("--get-mean-std", action='store_true')
        
        
        # training settings
        parser.add_argument("--use-awl", action='store_true')
        parser.add_argument("--warmup", action='store_false')
        parser.add_argument("--warmup-ratio", type=float, default=0.6)
        parser.add_argument("--warmup-epoch", type=int, default=1)
        parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
        parser.add_argument("--epochs", default=145, type=int, metavar="N", help="number of total epochs to run")
        parser.add_argument("--opt", type=str, default='sgd')
        parser.add_argument(
            "--lr",
            default=0.02,
            type=float,
            help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=str,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument(
            "--lr-scheduler", default="multi", type=str, help="name of lr scheduler (default: multisteplr)"
        )
        parser.add_argument(
            "--step-size", default=5, type=int)
        parser.add_argument(
            "--lr-steps",
            default=[8, 11],
            nargs="+",
            type=int,
            help="decrease lr every step-size epochs (multisteplr scheduler only)",
        )
        parser.add_argument(
            "--gamma", default=0.1, type=str, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
        )
        parser.add_argument(
            "--sync-bn",
            dest="sync_bn",
            help="Use sync batch norm",
            action="store_true",
        )
        parser.add_argument("--resume", action='store_true')
        parser.add_argument("--resume-tmp", action='store_true')
        parser.add_argument("--resume-file", default=None, type=str)
        parser.add_argument("--return-count", action='store_true',
                            help="return reloaded count for loss balancing on overloaded dataset")
        parser.add_argument("--grad-clip-value", default=None, type=int)
        parser.add_argument("--grad-to-none", action='store_true')
        
        
        # model settings
        parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
        parser.add_argument("--freeze-backbone", action='store_true')
        parser.add_argument(
            "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
        )
        # parser.add_argument("--model-type", default='general')
        parser.add_argument("--train-allbackbone", action='store_true')
        parser.add_argument("--freeze-bn", action='store_true')
        parser.add_argument("--dilation-type", default='fft', type=str)
        parser.add_argument("--use-neck", action='store_true')
        parser.add_argument("--approach", default='baseline')
        
        
        # classification settings
        parser.add_argument("--no-hflip", action='store_true')
        parser.add_argument("--loss-reduction-rate", default=0.9, type=float)
        
        
        # detection settings
        parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
        parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
        parser.add_argument("--convert_last_layer", action='store_true',
                            help='convert last layer from fc to conv or vise versa in only detector')
        
        
        ## segmentation
        parser.add_argument("--seg-bs", default=4, type=int)
        parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
        
        
        # evaluation settings
        parser.add_argument(
            "--validate",
            help="validate the model using validation datasets",
            action="store_true",
        )
        parser.add_argument(
            "--validate-only",
            action="store_true",
        )
        parser.add_argument("--get-shape", action="store_true")
        parser.add_argument("--get-flops", action="store_true")
        parser.add_argument("--load-ckpt", action="store_true")
        
        parser.add_argument("--make-cm", action="store_true")
        
        parser.add_argument("--find-epoch", default=None, type=int, help="the epoch for finding proper hyper parameters")
        
        # retrain settings
        parser.add_argument("--load-trained", default=None)
        parser.add_argument("--get-task_param", action='store_true')
        
        
        # environment settings
        parser.add_argument("--exp-case", default="0", type=str, help="exp case")
        parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
        parser.add_argument("--output-dir", default="/root/volume/exp", type=str, help="path to save outputs")
        
        # distributed training settings
        parser.add_argument("--distributed", action='store_true')
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
        parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
        parser.add_argument("--seed", default=0, type=int)
        # parser.add_argument("--distributed", action='store_false')

        parser.add_argument("--effect-type", default=None,)
        parser.add_argument("--effect-direction", default=None,)

        # Mixed precision training parameters
        parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
        
        
        # class_logit parser
        parser.add_argument("--filter", type=str, default=None)
        
        
        # others (추후에 정리할 것)
        parser.add_argument("--flop-size", nargs='+', type=int, default=[])
        parser.add_argument("--bottleneck-test", action='store_true')
        parser.add_argument("--loss-alpha", type=float, default=None)
        parser.add_argument("--alpha-task", nargs='+', type=str, default=None)
        parser.add_argument("--prototype", action='store_true')
        
        args = parser.parse_args()
        
        return args
    
    
class InferenceParser:
    def __init__(self) -> None:
        self.args = self._make_parser()
        
    def _make_parser(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--yaml_cfg", default=None)
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--save_name", default="result")
        parser.add_argument("--cfg", default=None)
        parser.add_argument("--infer_task", default=None, type=str)
        parser.add_argument("--infer_dataset", default=None, type=str)
        # parser.add_argument("--task", default=None)
        # parser.add_argument("--image-path", default=None)
        # parser.add_argument("--output-dir", default='/root/volume/exp')
        
        args = parser.parse_args()
        
        return args
    