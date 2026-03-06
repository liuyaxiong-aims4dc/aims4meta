# MSFlow: De novo molecular structure elucidation from mass spectra via flow matching
This is the codebase for our preprint: [MSFlow: De novo molecular structure elucidation from mass spectra via flow matching](https://arxiv.org/pdf/2602.19912).
For running the repo please follow the instructions:
## Environment installiation:
   - Install conda/miniconda if needed
   - Use [flow.yml](flow.yml) to create the necessary conda environment for using this codebase:
```
    conda env create -f flow.yml
    conda activate flow
```
## Data download/preprocessing
- To download data used for training MSFlow, please follow the same steps for download/preprocessing of data as illustrared in the repository [DiffMS](https://github.com/coleygroup/DiffMS). You need to clone DiffMS repository into the [ms_scripts](ms_scripts) directory for obtaining identical train/validation and also test sets.
- Then, you can derive CDDD representations for all datasets as illustrated in the repository [CDDDs](https://github.com/jrwnter/cddd)
### Encoder training:
- You can use CANOPUS and MassSpyGym training and validation data for training MS-CDDD encoder.
- You can check the original repository for retraining MIST using the provided script [train_mist.py](https://github.com/samgoldman97/mist/blob/main_v2/src/mist/train_mist.py) but with CDDD representations.

### Decoder training:
- After downloading the necessary training data, you can use [convert_smiles_to_safe.py](convert_smiles_to_safe.py) script for pre-processing decoder training and validation datasets and converting smiles into SAFE representation.
- For training the flow decoder, you can run [cfg_pretrain.py](cfg_pretrain.py). You will need to set the paths in [config.py](configs/data.py) to match the data directory.

## Inference with model weights
We  provide weights for our encoder-decoder pipeline for running inference [here](https://drive.google.com/drive/folders/18YovWipLFmO8-ziM4EyzWI5QmrMHzW1A?usp=drive_link).
- For MS-to-CDDD inference:  Inside the directory [ms_scripts](ms_scripts) you need to clone DiffMS repo, install the environment and the repo as a package and download the benchmarks following the authors instructions listed until preprocessing/downloading NPLIB1 and MSG benchmarks. We advice to create a seperate conda environment for encoder inference following the authors instructions. Then you can use [condition_inference.py](ms_scripts/condition_inference.py) script to run inference with our provided checkpoints and save MS embeddings to an output dataframe.
- Additionally, we provide some examples for running decoder inference using [inference.py](inference.py) that can be used after downloading the checkpoint and storing it in the existing checkpoints placeholder directory.

## License

MSFlow is released under the MIT license.

## Contact
If you have any inquiries, please reach out to ghaith.mqawass@tum.de
