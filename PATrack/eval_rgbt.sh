# test lasher
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name patrack --dataset_name LasHeR --yaml_name rgbt --epoch 58
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name patrack --dataset_name LasHeR --yaml_name rgbt --epoch 60

