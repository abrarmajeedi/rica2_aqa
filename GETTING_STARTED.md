
# Getting Started

  

## Requirements

Please find the installation instructions in [INSTALL.md](./INSTALL.md)

## Code
Pull the code using 

    git clone https://github.com/abrarmajeedi/rica2_aqa
 
 and navigate into the code directory
 

    cd rica2_aqa
  

## Data

You can download the zipped data from the Google drive [link](https://drive.google.com/file/d/1CjYtxnjHZzDkWDYrLMFbph9b-EZ8fdFT/view?usp=sharing).


Once downloaded, unzip the archive into ./data into the code directory


Make sure the data follows this structure
```markdown
├── data
│
│ ├── finediving
│ │ ├── Annotations
│ │ │ ├── Annotation files (**.pkl)
│
│ │ ├── FINADiving_MTL_256
│ │ │ ├── Video Frame directories
│
│ ├── mtl_aqa
│ │ ├── frames_long
│ │ │ ├── Video frame directories
│
│ │ ├── info
│ │ │ ├── Annotation files (**.pkl)
```

## Pretrained I3D weights

You can download the pretrained I3D weights from the Google drive [link](https://drive.google.com/file/d/1vi-C3V_i4Sy_4Y3yJLLeiGRzpz8Evvid/view?usp=sharing).


Once downloaded, place the file in `./pre_trained/model_rgb.pth`



## Running the code

Use the following commands to run the code

### FineDiving

    python -u train.py configs/fine/stoch_fine_diving_text_data_query.yaml

To run the deterministic  RICA<sup>2</sup>†

    python -u train.py configs/fine/deter_fine_diving_text_data_query.yaml
 
### MTL-AQA

    python -u train.py configs/mtl_aqa/stoch_mtl_diving_text_data_query.yaml

To run the deterministic  RICA<sup>2</sup>†

    python -u train.py configs/mtl_aqa/deter_mtl_diving_text_data_query.yaml


These commands will train the specified models and automatically run the evaluation, generating the evaluation results at the end.


## [BONUS] Tuning Experiment Parameters

Our code allows easy change of model and experiment parameters:

### Modifying  hyperparameters

You can modify different hyperparameters of the models and training by changing values within in the config files in `./configs`


### Generating text embeddings

RICA<sup>2</sup> incorporates the step information in the an action via LLM embedddings extracted from the textual step descriptions. These can be found `./tools`.

For FineDiving

    python ./tools/finediving/finediving_t5xxl_text_embed_extraction.py

For MTL-AQA

    python ./tools/mtl_aqa/mtl_t5xxl_text_embed_extraction.py


You can easily change the models used for extracting embeddings from the step descriptions by following the user-friendly [HuggingFace documentation](https://huggingface.co/docs).


## Contact

Email: majeedi+at+wisc+dot+edu