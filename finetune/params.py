import argparse


def get_finetune_params():

    parser = argparse.ArgumentParser(description='Finetune model on downstream tasks')

    # task settings
    parser.add_argument('--task_cfg_path',  type=str, default='', help='Path to the task configuration file')
    parser.add_argument('--exp_name',       type=str, default='', help='Experiment name')
    parser.add_argument('--pat_strat',      action='store_true', default=False, help='Patient stratification')

    # input data settings
    parser.add_argument('--dataset_csv',    type=str, default='', help='Dataset csv file')
    parser.add_argument('--split_dir',      type=str, default='data_split', help='Split directory')
    parser.add_argument('--pre_split_dir',  type=str, default='', help='Specify the pre-split directory, if it is specified, we will skip automatic split')
    parser.add_argument('--root_path',      type=str, default='', help='The tile encodings path')
    parser.add_argument('--tile_size',      type=int, default=256, help='Tile size in pixels')
    parser.add_argument('--max_wsi_size',   type=int, default=250000, help='Maximum WSI size in pixels for the longer side (width or height).')

    # model settings
    parser.add_argument('--model_arch',     type=str, default='vit_base')
    parser.add_argument('--input_dim',      type=int, default=1536, help='Dimension of input tile embeddings')
    parser.add_argument('--latent_dim',     type=int, default=768, help='Hidden dimension of the slide encoder')
    parser.add_argument('--feat_layer',     type=str, default='11', help='The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers')
    parser.add_argument('--pretrained',     type=str, default='/ruiyan/yuhao/project/FMBC/ibot/checkpoint0040.pth', help='Pretrained GigaPath slide encoder')
    parser.add_argument('--freeze',         action='store_true', default=False, help='Freeze pretrained model')
    parser.add_argument('--global_pool',    action='store_true', default=False, help='Use global pooling, will use [CLS] token if False')

    parser.add_argument('--seed',           type=int, default=0, help='Random seed')
    parser.add_argument('--epochs',         type=int, default=30, help='Number of training epochs')
    parser.add_argument('--warmup_epochs',  type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--batch_size',     type=int, default=1, help='Current version only supports batch size of 1')
    parser.add_argument('--lr',             type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--blr',            type=float, default=4e-3, help='Base learning rate, will caculate the learning rate based on batch size')
    parser.add_argument('--min_lr',         type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--lr_scheduler',   type=str, default='cosine', help='Learning rate scheduler', choices=['cosine', 'fixed'])
    parser.add_argument('--gc',             type=int, default=32, help='Gradient accumulation')
    parser.add_argument('--folds',          type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--optim',          type=str, default='adamw', help='Optimizer', choices=['adam', 'adamw'])
    parser.add_argument('--optim_wd',       type=float, default=0.05, help='Weight decay')
    parser.add_argument('--layer_decay',    type=float, default=0.95, help='Layer-wise learning rate decay')
    parser.add_argument('--dropout',        type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--val_r',          type=float, default=0.0, help='Ratio of data used for validation')
    parser.add_argument('--model_select',   type=str, default='val', help='Criteria for choosing the model checkpoint', choices=['val', 'last_epoch'])
    parser.add_argument('--save_dir',       type=str, default='outputs', help='Save directory')
    parser.add_argument('--num_workers',    type=int, default=10, help='Number of workers')
    parser.add_argument('--report_to',      type=str, default='wandb', help='Logger used for recording', choices=['wandb', 'tensorboard'])
    parser.add_argument('--fp16',           action='store_true', default=True, help='Fp16 training')
    parser.add_argument('--weighted_sample',action='store_true', default=False, help='Weighted sampling')
    
    ## MIL Model Settings
    parser.add_argument('--mil_model_size',   type=str, default='small', help='Class weight')
    parser.add_argument('--mil_type',    type=str, default='clam_sb', help='Number of workers')
    parser.add_argument('--pretrain_model', default='FMBC',help='type of pretrain model, ctans, UNI, CONCH, CHIEF,etc ... ')
    parser.add_argument('--pretrain_model_type', default='slide_level',help='type of pretrain model, patch level or slide level ')
    parser.add_argument('--tuning_method',)
    parser.add_argument('--experiment', type=str, default='finetune', help='Experiment name')
    parser.add_argument('--return_all_tokens', action='store_true', default=True, help='Return all tokens')
    parser.add_argument('--pool_method',type=str, default='cls_token', help='Return all tokens')
    
    ## Fintuning Settings
    #choice linear probe or MIL,only two class
    parser.add_argument('--tuning_method', type=str, options=['linear_probe', 'mil'], default='linear_probe', help='Tuning method')

    return parser.parse_args()
