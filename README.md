# PolyCLOVER: Polymer CombinatoriaL Optimization Via machinE leaRning

This is the official implementation of **PolyCLOVER**, a framework for polymer combinatorial optimization leveraging machine learning techniques.

## System requirements
This package has been tested on the following system:
- Linux: Ubuntu 22.04
- CUDA 12.4

## Environment configuration
Clone this repository:
```
git clone https://github.com/wuyuhui-zju/PolyCLOVER.git
```
 Create the environment with Anaconda in few minutes:
```
cd PolyCLOVER
conda env create
```
Activate the created enviroment:
```
conda activate polyclover
```

## Project structure
The major folders and their functions in PolyCLOVER are organized as follows:
```
PolyCLOVER/
├── datasets/       # Datasets for pretraining, finetuning, and evaluation
├── models/         # Pre-trained and fine-tuned model checkpoints
├── results/        # Optimization outputs
├── scripts/        # Scripts for preprocessing, training, optimization, and evaluation
├── src/            # Source code of PolyCLOVER (model architectures, training pipelines, utils)
├── environment.yml # Conda environment file
├── README.md       # Project introduction and usage guide
```

## Quick start
You can download the pre-trained model checkpoints and preprocessed datasets for a quick start.
Please place the downloaded `datasets/` and `models/` directories directly under the project root.

If you use the downloaded files, you can skip **Library preprocessing**, **Self-supervised learning** and directly proceed to Step 3: **Ensemble predictor training**.


## 1. Library preprocessing
We first preprocess the raw combinatorial library data. Run the following scripts to process the datasets:
```
python preprocess_raw_dataset.py  --dataset database --label False
python preprocess_database.py
```
- `preprocess_raw_dataset.py`: Handles the initial raw data formatting.
- `preprocess_database.py`: Constructs graphs for pretraining and prediction.

## 2. Self-supervised learning
We provide a shell script to pre-train the model:
```
bash pretrain.sh
```
Inside the script, you can specify options:
- `--train_mode`: Choose to train from scratch or load from the pre-trained graph encoder

## 3. Ensemble predictor training
Preprocess the labeled dataset:
```
python preprocess_raw_dataset.py --dataset initial
python preprocess_downstream_dataset.py --dataset initial
python split_dataset.py --dataset initial
```
- `--dataset`: Specify the dataset used for fine-tuning (such as `initial`, `round1`, `round2`,...).

After preprocessing, fine-tune the model on downstream tasks:
```
bash finetune.sh
```

Example of manual run:
```
for seed in $(seq 0 19)
do
  python finetune.py \
    --model_path ../models/pretrained/finetuned/base.pth \
    --dataset initial \
    --weight_decay 1e-6 \
    --dropout 0 \
    --lr 5e-4 \
    --save \
    --ensemble_idx $seed \
    --seed $seed
done
```
Inside the script, you can specify hyperparameters such as:
- `--lr`, `--dropout`, `--weight_decay`: Optimized learning rate, dropout ratio, weight decay.

## 4. Multi-objective optimization
To perform Bayesian optimization-based active learning:
```
bash mobo.sh
```
Inside the script, you can specify options:
- `--model_path`: Select the path of the fine-tuned predictor.
- `--dataset`: Dataset used.

The generated recommendation samples are saved in the `results` directory.

## 5. Web-lab synthesis and labeling
After a round of active learning, top candidates can be experimentally synthesized and labeled.  Merge the new data with the original data and place it in `datasets/round1/round1_raw.csv`
Then repeat the steps of **Ensemble predictor training**, **Multi-objective optimization**, and **Web-lab synthesis and labeling**.

## 6. Model evaluation
Please download the preprocessed file and place it in the `datasets/eval_antibacterial/` and `datasets/eval_hemolytic/` directory.
Evaluate the test performance:
```
bash eval_model.sh
```

## Citation
If you find PolyCLOVER helpful, please cite our work. (The official paper will be available soon.)
