# from gigapath.pipeline import tile_one_slide
# import huggingface_hub
# import glob
# import os
# from functools import partial
# from multiprocessing import Pool

# # Set your Hugging Face API token
# os.environ["HF_TOKEN"] = "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"
# assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

# def process_slide(slide, save_dir):
#     """Process a single slide."""
#     tile_one_slide(slide, save_dir=save_dir, level=1)
#     print(f'slide {slide} has been tiled')

# if __name__ == '__main__':
#     dataset = 'TCGA-LUAD'
#     raw_dir = '/home/yuhaowang/data/raw_data'
#     output_dir = '/home/yuhaowang/data/processed_data'
#     save_dir = os.path.join(output_dir, dataset)
#     slide_dir = os.path.join(raw_dir, dataset)
#     slide_list = glob.glob(os.path.join(slide_dir, '*/*.svs'))

#     # Use multiprocessing Pool to parallelize processing of slides
#     with Pool(processes=45) as pool:
#         pool.map(partial(process_slide, save_dir=save_dir), slide_list)

#     print('All slides have been tiled')

from gigapath.pipeline import tile_one_slide
import huggingface_hub
import glob
import os
from functools import partial
from multiprocessing import Pool

# Set your Hugging Face API token
os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

def process_slide(slide, save_dir):
    """Process a single slide."""
    tile_one_slide(slide, save_dir=save_dir, level=1)
    print(f'slide {slide} has been tiled')

if __name__ == '__main__':
    dataset = 'TCGA-THCA'
    raw_dir = '/home/yuhaowang/data/raw_data'
    output_dir = '/home/yuhaowang/data/processed_data'
    save_dir = os.path.join(output_dir, dataset)
    slide_dir = os.path.join(raw_dir, dataset)
    slide_list = glob.glob(os.path.join(slide_dir, '*/*.svs'))

    # Use a larger pool size to maximize CPU usage
    num_processes = 20

    # Use multiprocessing Pool to parallelize processing of slides
    with Pool(processes=num_processes) as pool:
        # Adjust chunksize based on the slide list length and number of processes
        chunksize = max(1, len(slide_list) // (num_processes * 4))
        pool.imap_unordered(partial(process_slide, save_dir=save_dir), slide_list, chunksize=chunksize)
        pool.close()
        pool.join()

    print('All slides have been tiled')


