# Learning_Time_Series_EHR

(to be finished)
## 1. Extract data from MIMIC-IV an eICU using METRE 
## 2. Run preprocessing notebook
## 3. Run prediction models 
> No fusion Transformer model

        python main.py  --model_name Transformer

> Fusion at I, V, VI

    python main.py  --static_fusion all --checkpoint test --num_channels 256 256 256 256 --s_param 256 256 256 0.2 --c_param 256 256 0.2 --sc_param 256 256 256 0.2
   
## 4. Run static info prediction models