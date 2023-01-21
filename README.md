# FPAC
We give an example of resnet50 on Imagenet.

1.run cen_generation.py to generate the distance of centroid deviation.

   python cen_generation.py --data_dir data_dir --resume ../resnet_50.pth --arch resnet_50 --adjust_ckpt --dataset imagenet  --limit 20

2.Determine the number of filters to be reserved, and then finetune the model.

   python evaluate.py --dataset imagenet --data_dir data_dir --job_dir ./result/resnet_50/folder_name --resume ../resnet_50.pth --arch resnet_50 --compress_rate [0.]+[0.1]*3+[0.35]*16 
