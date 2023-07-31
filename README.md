# Learning_Time_Series_EHR

(The repo to extract the data is in [METRE](https://github.com/weiliao97/METRE) and the repo to seperate sensitive latents and non-sensitive latents is in [TSVAE](https://github.com/weiliao97/TSVAE))

(to be finished)

## 1. Extract data from MIMIC-IV an eICU using METRE 
To extract data from MIMIV-IV and eICU databases, please follow the instructions in the [METRE](https://github.com/weiliao97/METRE) repository. To obtain the General cohort, please run:

    python main.py --database MIMIC --project_id xxx
   
  and 
  
     python main.py --database eICU --project_id xxx

MIMIC extraction has to be run before eICU due to eICU data normalization uses extracted MIMIC-IV data statistics. 

To obtain the Sepsis_3 cohort, please run:

    python main.py --database MIMIC --project_id xxx --patient_group Sepsis_3

And 

    python main.py --database eICU --project_id xxx --patient_group Sepsis_3


## 2. Run preprocessing notebook
## 3. Run prediction models 
> No fusion Transformer model

        python main.py  --model_name Transformer
> Fusion at I, V, VI

    python main.py  --static_fusion all --checkpoint test --num_channels 256 256 256 256 --s_param 256 256 256 0.2 --c_param 256 256 0.2 --sc_param 256 256 256 0.2
   
   ## 4. Run static info prediction models
