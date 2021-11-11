# 3DeepCT

## Setup
### Requirements
* Anaconda

Start a clean virtual environment:

```
conda create -n 3DeepCT python=3.7.9
conda activate 3DeepCT
```

Install required packages:
```
conda install --yes -c pytorch pytorch=1.10.0 torchvision cudatoolkit=<CUDA_VERSION>
pip install -r requirements.txt
```

## Usage
To train or test the model it is required to have a directory for your experiment under the experiments dir containing config.json file.
You can see an example in 3DeepCT\experiments\example.
 ### Training:
 
 `python train.py --exp_dir {EXP_DIR}`
 
 While training the model will be saved in {EXP_DIR} every {save_model_gap} epochs (from config.json file) under the name model_{epoch}.
 
 ### Evaluation:
 
 `python eval.py --exp_dir {EXP_DIR}`
 
 Evaluation will test the model specified in {model_path} from the config.json file.
 At the end of the evaluation, an evaluation_result.mat file will be saved in {EXP_DIR}.

# Citation
If you make use of our work, please cite our paper:

```
@inproceedings{sde20213deepct,
  title={3DeepCT: Learning Volumetric Scattering Tomography of Clouds},
  author={Sde-Chen, Yael and Schechner, Yoav Y and Holodovsky, Vadim and Eytan, Eshkol},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5671--5682},
  year={2021}
}
```
