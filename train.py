import argparse,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import logging
import importlib
from trainer import Trainer


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# [segnet, fcn, deconvnet, fpn, deeplab_v3, segformer]
def main():    
    # Load options
    parser = argparse.ArgumentParser(description='Attribute Learner')
    # parser.add_argument('--config', type=str, default="configs.segformer_base" 
    parser.add_argument('--config', type=str, default="configs.unet_base" 
    # parser.add_argument('--config', type=str, default="configs.pspnet_base" 
    # parser.add_argument('--config', type=str, default="configs.deeplab_v3_base" 
    # parser.add_argument('--config', type=str, default="configs.deeplab_v3_plus_base" 

    # parser.add_argument('--config', type=str, default="configs.segnet_base" 
    # parser.add_argument('--config', type=str, default="configs.fcn_base" 
    # parser.add_argument('--config', type=str, default="configs.deconv_base" 
    # parser.add_argument('--config', type=str, default="configs.fpn_base" 

                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument("--local_rank", type=int, help="local_rank")    
    parser.add_argument("--distributed", type=bool, default=False, help="distributed")                       
    
    conf = parser.parse_args() 
   
    opt = importlib.import_module(conf.config).get_opts()
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    logging.info('===Options==') 
    d=vars(opt)

    with open(os.path.join(d["out_path"], 'commandline_args.txt'), 'w') as f:        
        for key, value in d.items():
            if key in ["train_lines", "val_lines"]: continue
            num_space = 25 - len(key)
            try:
                f.write(key + " = " + str(value) + "\n")
            except Exception as e :
                pass

    for key, value in d.items():
        if key in ["train_lines", "val_lines"]: continue
        num_space = 25 - len(key)
        try:
            logging.info(": " + key + " " * num_space + str(value))
        except Exception as e:
            print(e)

    # Fix seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    cudnn.benchmark = True
    
    # Create working directories
    try:
        logging.info( 'Directory {} was successfully created.'.format(opt.out_path))
                   
    except OSError:
        logging.info( 'Directory {} already exists.'.format(opt.out_path))
        pass

    # Training
    t = Trainer(opt)
    t.train()

    print()

if __name__ == '__main__':
    main()    

    # delete __pycache__ 
    # find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

    # conda activate leyan_torch
    # CUDA_VISIBLE_DEVICES=6,7 python train.py
    # CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train.py

    # conda activate leyan_torch
    # python -m torch.distributed.launch --nproc_per_node=8 train.py