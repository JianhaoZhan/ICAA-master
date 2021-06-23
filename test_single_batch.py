import argparse
import os
'''python test_single_batch.py --rgb  --class_num  101  --gpu 0 --split 1 --base_dir '''

def parse_opts():
    parser = argparse.ArgumentParser()
    # Datasets 
    parser.add_argument(
        '--class_num',
        type=int)
    parser.add_argument(
        '--gpu',
        type=int)
    parser.add_argument(
        '--split',
        type=int)
    parser.add_argument(
        '--base_dir',
        type=str)
    parser.add_argument(
        '--rgb',
        action='store_true')
    args = parser.parse_args()
    return args


opt=parse_opts()
if opt.gpu==0:
    start=11
    end=16
elif opt.gpu==1:
    start=16
    end=21
elif opt.gpu==2:
    start=21
    end=26
if opt.class_num==51:
    if opt.rgb:
        for i in range(start,end):
            os.system('CUDA_VISIBLE_DEVICES={} python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split {} --only_RGB   --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1  '.format(opt.gpu,opt.split) + opt.base_dir + 'varLR{}.pth'.format(i) + '> '+ opt.base_dir + '{}.txt'.format(i))
    else:
        for i in range(start,end):
            os.system('CUDA_VISIBLE_DEVICES={} python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality Flow --sample_duration 64 --split {}  --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1 '.format(opt.gpu,opt.split) + opt.base_dir + 'varLR{}.pth'.format(i) + '> '+ opt.base_dir + '{}.txt'.format(i))
elif opt.class_num==101:
    if opt.rgb:
        for i in range(start,end):
            os.system('CUDA_VISIBLE_DEVICES={} python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split {} --only_RGB   --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 '.format(opt.gpu,opt.split) + opt.base_dir + 'varLR{}.pth'.format(i) + '> '+ opt.base_dir + '{}.txt'.format(i))
    else:
        for i in range(start,end):
            os.system('CUDA_VISIBLE_DEVICES={} python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality Flow --sample_duration 64 --split {}  --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 '.format(opt.gpu,opt.split) + opt.base_dir + 'varLR{}.pth'.format(i) + '> '+ opt.base_dir + '{}.txt'.format(i))




    
