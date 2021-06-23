TRAIN:

checkpoint:
python finetune.py --dataset UCF101 --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 101 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --pretrain_path "trained_models/Kinetics/RGB_Kinetics_64f.pth" --result_path "results/test/ucf/rgb" --checkpoint 1 --n_epochs 25 --resume_path1 '/home/eed-server3/Desktop/DATA4T/mars/MARS-master/final_model/split1/ucf/rgb/22_0.96.pth'


HMDB-RGB
python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/RGB_Kinetics_64f.pth" --result_path "results/test/split3/hmdb_rgb" --checkpoint 1 --n_epochs 25

HMDB-FLOW
python train.py --dataset HMDB51 --modality Flow --split 1 --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/Flow_Kinetics_64f.pth" --result_path "results/test/split3/hmdb_flow" --checkpoint 1 --n_epochs 25

HMDB-MARS
python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1 --n_classes 400 --n_finetune_classes 51 --batch_size 15 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --output_layers 'avgpool' --MARS_alpha 50 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/MARS_Kinetics_64f.pth" --checkpoint 1  --n_epochs 25  --result_path "results/test/split1/hmdb_mars" --resume_path1 

UCF-RGB
python train.py --dataset UCF101 --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 101 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --pretrain_path "trained_models/Kinetics/RGB_Kinetics_64f.pth" --result_path "results/test/split3/ucf_rgb" --checkpoint 1 --n_epochs 22

UCF-FLOW
python train.py --dataset UCF101 --modality Flow --split 1 --n_classes 400 --n_finetune_classes 101 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --pretrain_path "trained_models/Kinetics/Flow_Kinetics_64f.pth" --result_path "results/final/split1/ucf_flow" --checkpoint 1 --n_epochs 25

UCF-MARS
python MARS_train.py --dataset UCF101 --modality RGB_Flow --split 1 --n_classes 400 --n_finetune_classes 101 --batch_size 15 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --output_layers 'avgpool' --MARS_alpha 50 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --pretrain_path "trained_models/Kinetics/MARS_Kinetics_64f.pth" --checkpoint 1  --n_epochs 25  --result_path "results/test/split1/ucf_mars" --resume_path1


TEST:


CUDA_VISIBLE_DEVICES=0 
HMDB-RGB
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split 1 --only_RGB   --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1

HMDB-FLOW
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality Flow --sample_duration 64 --split 1  --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1 

HMDB two_stream
python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 64 --split 1  --frame_dir "dataset/HMDB51/" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1  --resume_path2 

HMDB MARS
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split 1 --only_RGB --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --result_path "results/"  --resume_path1 

UCF-RGB
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split 1 --only_RGB   --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 

UCF-FLOW
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality Flow --sample_duration 64 --split 2  --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 

UCF two_stream
python test_two_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB_Flow --sample_duration 64 --split 1  --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1  --resume_path2

UCF MARS
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split 1 --only_RGB --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --result_path "results/"  --resume_path1


python finetune.py --dataset HMDB51 --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/RGB_Kinetics_64f.pth" --result_path "results/final_model/split1/hmdb/rgb" --checkpoint 1 --n_epochs 5 --resume_path1 

python finetune.py --dataset HMDB51 --modality Flow --split 1 --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/Flow_Kinetics_64f.pth" --result_path "results/final_model/split1/hmdb/flow" --checkpoint 1 --n_epochs 5 --resume_path1 

python finetune.py --dataset UCF101 --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 101 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels" --pretrain_path "trained_models/Kinetics/RGB_Kinetics_64f.pth" --result_path final_model/split1/ucf/rgb/ --checkpoint 1 --n_epochs 5 --resume_path1


python train.py --dataset HMDB51 --modality Flow --split 1 --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/Flow_Kinetics_64f.pth" --result_path "results/final/split1/hmdb_flow" --checkpoint 1 --n_epochs 25 && python train.py --dataset HMDB51 --modality Flow --split 3 --n_classes 400 --n_finetune_classes 51 --batch_size 30 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 4 --frame_dir "dataset/HMDB51" --annotation_path "dataset/HMDB51_labels" --pretrain_path "trained_models/Kinetics/Flow_Kinetics_64f.pth" --result_path  results/final/split3/hmdb_flow/ --checkpoint 1 --n_epochs 25















final test:
ucf101:
rgb:
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split 1 --only_RGB   --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/UCF101 --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 ./final_model/split1/ucf/rgb/varLR4.pth
flow:
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality Flow --sample_duration 64 --split 1  --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/UCF101 --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 ./final_model/split1/ucf/flow/varLR22.pth
two:
python test_two_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB_Flow --sample_duration 64 --split 1  --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/UCF101 --annotation_path "dataset/UCF101_labels" --result_path "results/" --resume_path1 ./final_model/split1/ucf/rgb/varLR4.pth  --resume_path2 ./final_model/split1/ucf/flow/varLR22.pth
mars:
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split 1 --only_RGB --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/UCF101 --annotation_path "dataset/UCF101_labels" --result_path "results/"  --resume_path1 ./final_model/split1/ucf/mars/varLR19.pth
three:
python test_three_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB_Flow --sample_duration 64 --split 1 --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/UCF101 --annotation_path "dataset/UCF101_labels/" --result_path "results/"   --resume_path1 ./final_model/split1/ucf/rgb/varLR4.pth  --resume_path3 ./final_model/split1/ucf/flow/varLR22.pth  --resume_path2 ./final_model/split1/ucf/mars/varLR20.pth

hmdb51
rgb:
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split 1 --only_RGB   --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/HMDB51 --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1 ./final_model/split1/hmdb/rgb/varLR22.pth 
flow:
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality Flow --sample_duration 64 --split 1  --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/HMDB51 --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1  ./final_model/split1/hmdb/flow/vaLR12.pth
mars:
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split 1 --only_RGB --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/HMDB51 --annotation_path "dataset/HMDB51_labels" --result_path "results/"  --resume_path1 ./final_model/split1/hmdb/mars/24_0.8085.pth

mars+rgb+flow
python test_three_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 64 --split 2 --frame_dir /home/eed-server3/Downloads/MARS/MARS-master/dataset/HMDB51 --annotation_path "dataset/HMDB51_labels/" --result_path "results/"   --resume_path1 results/final/split2/hmdb_rgb/varLR13.pth   --resume_path3 final_model/split2/hmdb/flow/varLR22.pth  --resume_path2 trained_models/HMDB51/MARS_HMDB51_2_64f.pth
