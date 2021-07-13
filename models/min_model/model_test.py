from __future__ import division
import torch
from torch import nn
from models import resnext_mul_serial # resnext_mul resnext_mul+para 
import pdb,apex

def generate_model( opt):
    assert opt.model in ['resnext']
    assert opt.model_depth in [101]

    
    model = resnext_mul_serial.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers,
            )

    from models.resnext_mul_serial import get_fine_tuning_parameters    
    parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)

    model = model.cuda()

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening
    optimizer = torch.optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    model = apex.parallel.convert_syncbn_model(model)
    model, optimizer = apex.amp.initialize(model, optimizer)
    model = apex.parallel.DistributedDataParallel(model)#, device_ids=[opt.local_rank])
    #model = nn.DataParallel(model)
    #model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[-1])
    #model = apex.parallel.DistributedDataParallel(model)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        
        # opt.arch = ResNext-101
        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'], False)
        # load parameters in pretrain model from 1000 to 101/51
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()
        
        # opt.ft_begin_index : Begin block index of fine-tuning : 4
        return model, parameters,optimizer

