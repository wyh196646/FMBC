
from gigapath.pipeline import tile_one_slide
import huggingface_hub
import glob
import os
from functools import partial
from multiprocessing import Pool
import shutil
# Set your Hugging Face API token
os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

def process_slide(slide, save_dir):
    """Process a single slide."""
    tile_one_slide(slide, save_dir=save_dir, level=1)
    print(f'slide {slide} has been tiled')
    
def valid_dir(dir_path):
    if len(os.listdir(dir_path)) <=3:
        shutil.rmtree(dir_path)
        print(f'{dir_path} is removed')
        
if __name__ == '__main__': 
    raw_dir = '/mnt/data/ruiyan/clearning'
    output_dir = '/mnt/data/ruiyan/processed_data'
    support_suffixs=['.svs','.tif','.tiff']
    #TCGA-BLCA  TCGA-BRCA  TCGA-COAD  TCGA-LUAD  TCGA-LUSC  TCGA-THCA
    #datasets=['BACH','Post-NAT-BRCA','Multi-omic','IMPRESS','HE-vs-MPM']
    #datasets=['CAMELYON17']
    datasets  = os.listdir(raw_dir)
    exclude_dirs = ['private_chunk_1','private_chunk_2','private_chunk_3','private_chunk_4',
                    'private_chunk_5',  'private_chunk_6','private_chunk_7','private_chunk_8',
                    'private_chunk_9','private_chunk_10']
    datasets = ['BACH']
    datasets= [dataset for dataset in datasets if dataset not in exclude_dirs]
    for dataset in datasets:
        print(f'We are now Processing {dataset}...')
        save_dir = os.path.join(output_dir, dataset) 
        slide_dir = os.path.join(raw_dir, dataset)
        slide_list=[]
        for suffix in support_suffixs:
            slide_list.extend(glob.glob(os.path.join(slide_dir, f'**/*{suffix}'),recursive=True))  
            
        if len(slide_list)==0:
            print(f'{dataset} has no slide')
            continue
        #print(slide_list)
        # Use a larger pool size to maximize CPU usage
        #slid_list=['/data1/BCOData/BRACS/BRACS_WSI/train/Group_AT/Type_FEA/BRACS_1858.svs']
        num_processes = 10

        with Pool(processes=num_processes) as pool:
            # Adjust chunksize based on the slide list length and number of processes
            chunksize = max(1, len(slide_list) // (num_processes * 4))
            pool.imap_unordered(partial(process_slide, save_dir=save_dir), slide_list, chunksize=chunksize)
            pool.close()
            pool.join()

        print('All slides have been tiled')
        
        for dir in os.listdir(os.path.join(save_dir,'output')):
            print(os.path.join(save_dir,'output', dir))
            valid_dir(os.path.join(save_dir,'output', dir))
            
        for slide_path in slide_list:
            try:
                print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
                tile_one_slide(slide_path, save_dir=save_dir, level=1)
                print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
            except Exception as e:
                print(f"Error processing slide {slide_path}: {e}")
