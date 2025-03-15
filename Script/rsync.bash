#!/bin/bash

# 定义3个 rsync 命令
cmd1="rsync -avz --progress --partial yuhaowang@172.16.120.21:/mnt/data/ruiyan/processed_data/{ACROBAT,BCNB,CAMELYON16,CPTAC-BREAST-all,HE-vs-MPM,Multi-omic,private_chunk_4,private_chunk_7,TIGER} /data1/yuhaowang"
cmd2="rsync -avz --progress --partial yuhaowang@172.16.120.21:/mnt/data/ruiyan/processed_data/{AIDPATH,BRACS,CAMELYON17,GEO-GSE243280,HyReCo,Post-NAT-BRCA,private_chunk_2,private_chunk_5,private_chunk_8,SLN-Breast,TUPAC,HIST2ST,private_chunk_10} /data2/yuhaowang"
cmd3="rsync -avz --progress --partial yuhaowang@172.16.120.21:/mnt/data/ruiyan/processed_data/{BACH,BreakHis,CMB-BRCA,GTEX_Breast,IMPRESS,private_chunk_1,private_chunk_3,private_chunk_6,private_chunk_9,TCGA-BRCA} /data3/yuhaowang"

# 使用 xargs 让3个任务并行执行，并实时输出日志
echo -e "$cmd1\n$cmd2\n$cmd3" | xargs -I CMD -P 3 bash -c "CMD | tee -a sync.log"

echo "所有同步任务已完成！"




         