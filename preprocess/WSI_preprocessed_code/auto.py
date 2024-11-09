#to auto run the preprocessing code

import os
import subprocess
import argparse
import os
import shutil
# 文件名和对应的命令行参数


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto run the preprocessing code')
    parser.add_argument('--orinal_storge_path',type=str,default='/data/wyh_data/data/raw_data',help='path to the WSI files')
    parser.add_argument('--storge_path', type=str,default='/home/yuhaowang/data/raw_data', help='path to the WSI files')
    parser.add_argument('--embedding_path', type=str, default='/home/yuhaowang/data/embedding',help='path to the mask files')
    args = parser.parse_args()
    dataset_list=os.listdir(args.orinal_storge_path)
    dataset_list=['TCGA-Toy']
    for dataset in dataset_list:
        #copy dataset in original storge path to storge path
        temp_raw_data_dir=os.path.join(args.storge_path,dataset)
        mask_dir=os.path.join(args.storge_path,dataset+"-Mask")
        patch_dir=os.path.join(args.storge_path,dataset+"-Patch")
        embedding_dir=os.path.join(args.embedding_path,dataset)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        if os.path.exists(embedding_dir):
            print('dataset {} has been processed'.format(dataset))
            continue
        if not os.path.exists(temp_raw_data_dir):
            shutil.copytree(os.path.join(args.orinal_storge_path,dataset),temp_raw_data_dir)
        scripts_and_args= [
        ("1tissue_mask_svs.py", [ "--wsi_path",temp_raw_data_dir,
                                "--mask_path", mask_dir]),
        
        ("2genPatch_multi.py", [  "--wsi_path", temp_raw_data_dir, 
                                "--mask_path", mask_dir, 
                                "--patch_path", patch_dir])
        ]
# 遍历每个文件并执行
        for script, args in scripts_and_args:
            #print("Running script:", script, "with args:", args)
            command = ["python", script] + args
            print("Command:", command)
            subprocess.run(command, check=True)
       
        #os.system("nohup torchrun --nproc_per_node=4 feature_extractor/get_features.py --data_path {} --dump_features {} >{}.log 2>&1 &".format
        #(patch_dir,embedding_dir,dataset+".log"))
        print(patch_dir)
        print(embedding_dir)
 
        
        # shutil.rmtree(temp_raw_data_dir)
        # shutil.rmtree(patch_dir)
        print(dataset,"is done")
   
