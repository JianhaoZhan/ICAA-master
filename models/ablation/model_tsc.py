from __future__ import division
import torch
from torch import nn
from models.ablation import resnext_tsc
import pdb 

def generate_model( opt):
    assert opt.model in ['resnext']
    assert opt.model_depth in [101]

    
    model = resnext_tsc.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers,
            )
    

    model = model.cuda()
    model = nn.DataParallel(model)
    
    if opt.pretrain_path:
        from models.ablation.resnext_tsc import get_fine_tuning_parameters
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        
        # opt.arch = ResNext-101
        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'], False)
        # load parameters in pretrain model from 1000 to 101/51
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()
        
        # opt.ft_begin_index : Begin block index of fine-tuning : 4
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()

