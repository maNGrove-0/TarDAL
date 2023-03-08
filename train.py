import argparse
import logging
from pathlib import Path

import torch.backends.cudnn
import yaml

import scripts
from config import from_dict

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    # 命令行输入yaml参数文件以及wandb key
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--auth', help='wandb auth api key')
    args = parser.parse_args()

    # init config
    # 将yaml文件中的数据加载到config中
    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)  # convert dict to object
    # 这一下是在干嘛呢
    config = config

    # init logger
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level=config.debug.log, format=log_f)

    # init device & anomaly detector
    torch.backends.cudnn.benchmark = True
    # 异常检测, 会在 backward 的时候检查NaN或其他异常的存在
    torch.autograd.set_detect_anomaly(True)

    # choose train script
    # 选择训练策略，有三种
    logging.info(f'enter {config.strategy} train mode')
    match config.strategy:
        # 针对人类视觉的优化
        case 'fuse':
            train_p = getattr(scripts, 'TrainF')
        # 下面的两种都是用的TrainFD的script
        # 针对目标检测进行的优化
        case 'detect':
            if config.loss.bridge.fuse != 0:
                logging.warning('overwrite fuse loss weight to 0')
                config.loss.bridge.fuse = 0
            train_p = getattr(scripts, 'TrainFD')
        # 同时针对人类视觉和检测准确度
        case 'fuse & detect':
            train_p = getattr(scripts, 'TrainFD')
        case _:
            raise ValueError(f'unknown strategy: {config.strategy}')

    # create script instance
    train = train_p(config, wandb_key=args.auth)
    train.run()
