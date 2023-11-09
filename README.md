# Learning_Time_Series_EHR

(The repo to extract the data is in [METRE](https://github.com/weiliao97/METRE) and the repo to seperate sensitive latents and non-sensitive latents is in [STEER](https://github.com/weiliao97/TIMESPAN))

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
### 3.1 48h in-hospital mortality task
In IHM folder, run

    python main.py --model_name TCN --num_channels 256 256 256 256 --checkpoint test 
### 3.2 SOFA score prediction 
In SOFA folder: 
> No fusion Transformer model

    python main.py  --model_name Transformer --checkpoint test

or 

    python main.py --model_name TCN --num_channels 256 256 256 256 --checkpoint test 
> Fusion at I, V and VI 

    python main.py  --static_fusion all --num_channels 256 256 256 256 --s_param 256 256 256 0.2 --c_param 256 256 0.2 --sc_param 256 256 256 0.2 --checkpoint test
> With regularization l1 or l2 

    python main.py --static_fusion inside --num_channels 256 256 256 256 --s_param 256 256 256 0.2 --c_param 256 256 0.2 --sc_param 256 256 256 0.2 --regularization l1 --checkpoint test 
   
## 4. Run static info prediction models
In infer_static folder, have the model pt file ready for feature extraction.
> TCN model with channels [256, 256, 256, 256], sepsis_3 cohort, SOFA prediction and infer race (index: 2):

    python main.py --model_path xxx --model_name TCN --num_channel 256 256 256 256 --task_name sofa --read_channels 128 --infer_ind 2


