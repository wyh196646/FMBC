mkdir -p /data4/processed_data

for dir in /data1/* /data2/* /data3/*; do
    target="/data4/processed_data/$(basename "$dir")"

    # 获取真实路径，避免创建循环
    real_path=$(readlink -f "$dir")
    
    # 避免创建指向自己的软链接
    if [[ "$real_path" == /data4/processed_data* ]]; then
        echo "⚠️ 警告: $dir 指向 /data4/processed_data，跳过"
        continue
    fi

    # 如果软链接已经存在，先删除它，防止错误
    if [ -L "$target" ]; then
        rm "$target"
    fi

    # 确保目标路径真实存在
    if [ -e "$real_path" ]; then
        ln -s "$real_path" "$target"
    fi
done
