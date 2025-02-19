import glob
import os
data_dir = '/ruiyan/yuhao/embedding/BCNB'
models = ['FMBC']
for model in models:
    h5_files = os.listdir(os.path.join(data_dir, model))
#rename h5 files with bcnb_ +
    for file in h5_files:
        filename = os.path.basename(file).split('.')[0]
        new_filename = os.path.join(data_dir,model, 'bcnb_'+filename+'.h5')
        #rint(file +'->'+new_filename)
        os.rename(os.path.join(data_dir,model,file), new_filename)
        
import glob
import os
data_dir = '/ruiyan/yuhao/embedding/TCGA-BRCA'
for model in models:
    h5_files = os.listdir(os.path.join(data_dir, model))
#rename h5 files with bcnb_ +
    for file in h5_files:
        filename = os.path.basename(file).split('.')[0].split('Z-00')[0]+'.h5'
        new_filename = os.path.join(data_dir,model, filename)
        print(os.path.join(data_dir,model, file+'.h5'))
        print(new_filename)
        os.rename(os.path.join(data_dir,model,file), new_filename)
        
