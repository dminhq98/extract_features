# extract_features

## requirement

annoy\
pytorch\
h5py\
tqdm\
Pillow

---
### 1. Etract feature 

**usage:** 
``` 
python extract_features.py [-h] --image-forder PATH [--model-name MODEL_NAME]\
                           --output-features PATH [--workers N]\
                           [--batch-size N] [--output_log OUTPUT_LOG]\
``` 
>**optional arguments:**\
  -h, --help            show this help message and exit
  --image-forder PATH   path to the image folder to extract (default: None)\
  --model-name MODEL_NAME
                        Name pretrained model to extract features from\
                        ,Default extract feature before layer fc. (default:
                        resnet50)\
  --output-features PATH
                        Output features as HDF5 to this location. (default:
                        None)\
  --workers N           number of data loading workers (default: 4)\
  --batch-size N\
  --output_log OUTPUT_LOG
                        Output file to log to. Default: --output_features +
                        ".log" (default: None)\
                     
                    
*Example:
``` 
python extract_features.py --image-forder dataset --model-name resnet50 --output-features features.h5 --workers 4 --batch-size 10
```
---
### 2. Index feature 
**usage:** 
``` 
python index_features.py [-h] --features-name PATH --output-index PATH\
                         [--output_log OUTPUT_LOG]\
``` 
>**optional arguments:**\
  -h, --help            show this help message and exit\
  --features-name PATH  File features as HDF5 from this location. (default:
                        None)\
  --output-index PATH   Output index as AnnoyIndex to this location. (default:
                        None)\
  --output_log OUTPUT_LOG
                        Output file to log to. Default: --output_features +
                        ".log" (default: None)\
                   
*Example :
``` 
python index_features.py --features-name features.h5 --output-index test.ann
```
---
### 3. Test result

**usage:** 
``` 
python test_retrival.py [-h] --test-forder PATH [--model-name MODEL_NAME]
                        --features-name PATH --index-name PATH
                        [--path-result PATH] [--output_log OUTPUT_LOG]\
``` 
>**optional arguments:**\
  -h, --help            show this help message and exit\
  --test-forder PATH    path to the image folder to test (default: None)\
  --model-name MODEL_NAME\
                        Name pretrained model to extract features from
                        ,Default extract feature before layer fc. (default:
                        resnet50)\
  --features-name PATH  File features as HDF5 from this location. (default:
                        None)\
  --index-name PATH     Index as AnnoyIndex to this location. (default: None)\
  --path-result PATH    Save result to this location. (default: result_test)\
  --output_log OUTPUT_LOG
                        Output file to log to. Default: --output_features +
                        ".log" (default: None)\

*Example :
``` 
python test_retrival.py --test-forder test_forder --model-name resnet50 --features-name features.h5 --index-name test.ann --path-result result_test
```

## Installation
First install conda , then:

```
conda create --name feature-extract python=3
source activate feature-extract
pip install requirements.txt
```


