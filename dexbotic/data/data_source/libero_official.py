from dexbotic.data.data_source.register import register_dataset

LIBERO_DATASET = {
    "goal": {
        "data_path_prefix": "./data/libero/libero_goal/video",
        "annotations": './data/libero/libero_goal',
        "frequency": 1,
    },
    "10": {
        "data_path_prefix": "./data/libero/libero_10/video",
        "annotations": './data/libero/libero_10',
        "frequency": 1,
    },
    "spatial": {
        "data_path_prefix": "./data/libero/libero_spatial/video",
        "annotations": './data/libero/libero_spatial',
        "frequency": 1,
    },
    "object": {
        "data_path_prefix": "./data/libero/libero_object/video",
        "annotations": './data/libero/libero_object',
        "frequency": 1,
    },
    "pi0_all": {
        "data_path_prefix": "./data/libero/libero_pi0_all/image",
        "annotations": './data/libero/libero_pi0_all',
        "frequency": 1,
    },
    "oft_all": {
        "data_path_prefix": "./data/libero/libero_oft_all",
        "annotations": './data/libero/libero_oft_all',
        "frequency": 1,
    }
}

meta_data = {
    'non_delta_mask': [6],
    'periodic_mask': None,
    'periodic_range': None
}

register_dataset(LIBERO_DATASET, meta_data=meta_data, prefix='libero')
