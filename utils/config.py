from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.data_path = ""  # Directory where datasets are stored
_C.descriptors_fname = ""
_C.num_class = 10
_C.class_names = []
_C.model = ""
_C.backbone = ""
_C.resolution = 224
_C.stride = 16

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.resume = None  # Path to a directory where the files were saved previously
_C.checkpoint_freq = 0  # How often (epoch) to save model during training. Set to 0 or negative value to only save the last one
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None
_C.deterministic = False
_C.gpuid = None
_C.num_workers = 8
_C.prec = "fp16"  # fp16, fp32, amp

_C.epochs = 10
_C.warmup = 0
_C.batch_size = 256
_C.lr = 0.03
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.lnl_methods = ""
_C.is_coop = False

_C.finetune = False
_C.bias_tuning = False
_C.bn_tuning = False  # only for resnet
_C.vpt_shallow = False
_C.vpt_deep = False
_C.vpt_len = 0
_C.adapter = False
_C.adapter_dim = 0
_C.lora = False
_C.lora_dim = 0
_C.ssf = False
_C.partial = None

_C.N_CTX = 16
_C.CTX_INIT = "a photo of a"
_C.CSC = False
_C.CLASS_TOKEN_POSITION = "end"

_C.zero_shot = False

_C.eval_only = False
_C.model_dir = None
_C.load_epoch = None

_C.noise_mode = ""
_C.noise_ratio = 0.0