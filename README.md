# Parallelized_LSM_for_Unintentional_Action_Recognition

- To train and validate the PLSM model, downlaod this repository.
- Download the Oops dataset and place the train and val folders inside PATH/TO/datasets/
- Run the basic_pipeline_v4.0.py script to generate data using the LSM.
- Make sure the paths to the dataset and annotations match.
- To downlaod the specific LSM model used in our work, download the model from https://drive.google.com/file/d/1F79GB-2PKKffvG2Ij5ylSUTvm5lQRUci/view?usp=sharing and transfer the model to location saved_models/
- Make sure you have created two empty folders named 'train' and 'val' inside MRI_data/PLSM_readout_100timesteps/
- Once the data is generated, run the script G_MRI_LR.py for training the 3D CNN readout layer.
- Make sure the datapaths to the generated data matches the one used in the G_MRI_LR.py script. 
