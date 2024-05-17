# Meme Captioning

## Introduction
This repository contains the code for the deep learning project focused on meme captioning.

## Installation
To replicate the environment and run the code follow these steps (**Warning**: a CUDA device must be used):

1. Clone the repository to your local machine 

    ```bash
    git clone https://github.com/Julian-JN/Meme-Captioning
    cd Meme-Captioning
    ```

2. Install dependencies:

    - Using pip
        ```bash
        pip install -r requirements.txt
      
    - WandB:
   
         To setup WandB logging, you must set the WandB key as an environment variable. An invitation link to the wandB log (INM706-Final) is provided here: https://wandb.ai/citai?invited=&newUser=false

## Usage

     - Checkpoints:

        Dowload checkpoints from here (same as dataset): 
        https://drive.google.com/drive/folders/1yxOv_ZH9PZ5hunWNYo509ORrOToGW9XB?usp=drive_link
        Please place the downloaded directory in the main 'Meme-Captioning' folder.


    - Dataset

        The dataset used in FLICKR and MEMECAP evaluations is available through:
        https://drive.google.com/drive/folders/1yxOv_ZH9PZ5hunWNYo509ORrOToGW9XB?usp=drive_link
        Please place the downloaded directories in the main 'Meme-Captioning' folder.

    - Adjust the settings in the 'config.yaml' file. 

        The model section of the config file allows you to select the encoder type (Resnet or EfficientNet),
        whether to have self-attention in the encoder or Bahdanau in the decoder through boolean variables.

        The train settings allows for configuration of certain hyperparameters, such as epochs, batch size, and learning rates for encoder and decoder.

    - Train
    - Important Note: in the dataset files for each dataset ('dataset.py', 'dataset_flickr.py'), PLEASE comment out the lines at the end of the get_item() function, in the return portion, which start with variable names "all_"

        To start the training run the 'train.py' file if you want to run meme captioining and 'train_flickr' if you want to run Flickr captioning: 
        ```bash
        python train.py
        ```

4. Inference and Visualisation of Attention

    - Change the checkpoint file name to be loaded in the 'memes_inference.py' file at lines 343-345:
    - Change the checkpoint file name to be loaded in the 'flickr_inference.py' file at lines 300-301:

- To change whether to plot self attention results or not, change in the test() instantiation the parameters plot_encoder_attention,
plot_decoder_attention to True or False. 
- To use a different metric, manually change lines with calculate_bleu()/calculate_meteor(). 
- Important Note: in the dataset files for each dataset ('dataset.py', 'dataset_flickr.py'), PLEASE uncomment out the lines at the end of the get_item() function, in the return portion, which start with variable names "all_"
- The checkpoints provided use EfficientNetb5 as the encoder, and have feature size of 2048 and attention sizes of 16*16
  - Visualise the results:

      Run:
      ```bash
      python memes_inference.py
      python flickr_inference.py
      ```