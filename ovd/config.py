from detectron2.config import CfgNode as CN


def add_ovd_config(cfg):
    _C = cfg

    # Open-vocabulary classifier
    _C.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False  # Use fixed classifier for open-vocabulary detection
    _C.MODEL.ROI_BOX_HEAD.WEIGHT_TRANSFER = False  # Use weight transfer layers in zero-shot classifier head
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_photo+cname.npy'
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    _C.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = 'datasets/lvis/lvis_v1_train_norare_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'pseudo_max_score'
    _C.MODEL.ROI_BOX_HEAD.PMS_LOSS_WEIGHT = 0.1  # pseudo-max score loss
    _C.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128
    _C.MODEL.ROI_HEADS.MASK_WEIGT = 1.0

    # Different classifiers in testing, used in cross-dataset evaluation
    _C.MODEL.RESET_CLS_TESTS = False
    _C.MODEL.TEST_CLASSIFIERS = []
    _C.MODEL.TEST_NUM_CLASSES = []

    # Configs specif to RKD and PIS
    _C.MODEL.PIS_PROP_PATH = ''
    _C.MODEL.RKD_FEAT_PATH = ''
    _C.MODEL.RKD_ILS_FEAT_PATH = ''
    _C.MODEL.DISTILLATION = False
    _C.MODEL.NUM_DISTIL_PROP = 5
    _C.MODEL.DISTIL_L1_LOSS_WEIGHT = 0.0
    _C.MODEL.IRM_LOSS_WEIGHT = 0.0

    # Multi-dataset dataloader
    _C.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    _C.DATALOADER.USE_RFS = [False, False]
    _C.DATALOADER.MULTI_DATASET_GROUPING = False  # Always true when multi-dataset is enabled
    _C.DATALOADER.DATASET_ANN = ['box', 'box']  # Annotation type of each dataset
    _C.DATALOADER.USE_DIFF_BS_SIZE = False  # Use different batchsize for each dataset
    _C.DATALOADER.DATASET_BS = [8, 32]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SIZE = [896, 384]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.5, 1.5)]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (320, 400)]  # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MAX_SIZES = [1333, 667]  # Used when USE_DIFF_BS_SIZE is on

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default'

    _C.FIND_UNUSED_PARAM = True
    _C.FP16 = False

    _C.SOLVER.OPTIMIZER = 'SGD'

    _C.WITH_IMAGE_LABELS = False  # Use Pseudo Image-level Supervision(PIS)
