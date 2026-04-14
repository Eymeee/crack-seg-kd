import kagglehub, os, shutil, random


def main(n_train:float = 0.70, n_val: float = 0.15): 
    path = kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset") + "/crack_segmentation_dataset"
    print("Path to dataset files:", path)
    src_images = f'{path}/images'
    src_masks  = f'{path}/masks'

    random.seed(42)
    files = sorted(os.listdir(src_images))
    random.shuffle(files)

    n       = len(files)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    splits = {
        'train' : files[:n_train],
        'val'   : files[n_train:n_train + n_val],
        'test'  : files[n_train + n_val:],
    }

    for split, subset in splits.items():
        os.makedirs(f'data/{split}/images', exist_ok=True)
        os.makedirs(f'data/{split}/masks',  exist_ok=True)
        for f in subset:
            mask_f = f if os.path.exists(f'{src_masks}/{f}') else f.replace('.jpg', '.png')
            shutil.copy(f'{src_images}/{f}',     f'data/{split}/images/{f}')
            shutil.copy(f'{src_masks}/{mask_f}', f'data/{split}/masks/{mask_f}')
        print(f'{split}: {len(subset)} images')



if __name__ == "__main__": 
    #this will download and partionate the data 
    main()