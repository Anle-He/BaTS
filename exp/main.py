import os
import sys
import yaml
import random
import argparse
import datetime
import numpy as np

import torch

sys.path.append('..')
from baselines import select_model


if __name__ == 'main':
    # --------- Setup running environment --------- #
    # Set random seed
    fix_seed = 2024

    os.environ['PYTHONHASHSEED'] = str(fix_seed)

    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed) # For multi-GPU

    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    """

    # Set cpu
    num_cpus = 3 # Limit the number of cpu threads
    torch.set_num_threads(num_cpus)

    os.environ['OMP_NUM_THREADS'] = str(num_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpus)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_cpus)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set args
    parser = argparse.ArgumentParser()
    ## Specify the model
    parser.add_argument(
        '-m',
        '--model_name',
        type=str,
        default='iTransformer'
    )
    ## Specify the task
    parser.add_argument(
        '-t',
        '--task_name',
        type=str,
        default='LTSF'
    )
    ## Specify the dataset
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        default='PEMS08'
    ) 
    ## Specify the config file path
    parser.add_argument(
        '-cfg',
        '--config_path',
        type=str,
        default='../baselines/iTransformer/LTSFConfig/PEMS08_IN96_OUT96.yaml'
    )
    args = parser.parse_args()

    model_arch = select_model(args.model_name)
    model_name = model_arch.__name__

    task_name = args.task_name.upper()

    dataset_name = args.dataset_name.upper()
    data_path = f'../data/{dataset_name}'

    cfg_path = args.cfg_path
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    
    # --------- Load the model --------- #
    # cfg.get(key, deault_value): No need to write in the config if not used.
    # cfg[key]: Must be assigned in the config, else KeyError.
    model = model_arch(**cfg['MODEL_PARAM']).to(DEVICE)