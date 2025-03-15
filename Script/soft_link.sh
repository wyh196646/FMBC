mkdir -p /data4/processed_data  # 创建统一访问的目录

# 遍历 /data1 里的数据集，创建软链接
for dir in /data1/yuhaowang/*; do
    ln -s "$dir" "/data4/processed_data/$(basename "$dir")"
done

# 遍历 /data2 里的数据集，创建软链接
for dir in /data3/yuhaowang/*; do
    ln -s "$dir" "/data4/processed_data/$(basename "$dir")"
done

# for dir in /data3/*; do
#     ln -s "$dir" "/home/yuhaowang/data/$(basename "$dir")"
# done

# 查看 /home/yuhaowang/data 目录下的内容
ls -l /data4/processed_data


#

#rsync -avz --progress --partial yuhaowang@172.16.120.34:/home/yuhaowang/project/FMBC/SlideModel /home/yuhaowang/project