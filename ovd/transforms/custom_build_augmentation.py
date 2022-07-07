from detectron2.data import transforms as T


def build_custom_augmentation(cfg, is_train, min_size=None, max_size=None):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge':
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN if min_size is None else min_size
            max_size = cfg.INPUT.MAX_SIZE_TRAIN if max_size is None else max_size
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    else:
        assert 0, cfg.INPUT.CUSTOM_AUG

    if is_train:
        augmentation.append(T.RandomFlip())
    return augmentation


build_custom_transform_gen = build_custom_augmentation
"""
Alias for backward-compatibility.
"""
