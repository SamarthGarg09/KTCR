# Knowledge Transfer-driven Concept Refinement for Hate Speech Detection

This repository implements Knowledge Transfer-driven Concept Refinement (KTCR), a novel approach for improving hate speech detection models' ability to recognize implicit hate patterns while maintaining performance on explicit content.

## Project Structure

```bash
├── src/
│   ├── autoencoder_module.py     # Autoencoder implementation for knowledge transfer
│   ├── cav_classifier_module.py  # Concept Activation Vector classifier
│   ├── student_module.py         # Student model with concept refinement
│   ├── config.yaml               # Configuration parameters
│   ├── utils.py                  # Utility functions for logging and data handling
├── training/
│   ├── train_student_model_wiki.py  # Student model training on Wiki dataset
│   ├── teacher_ft_ea.py            # Teacher fine-tuning on EA dataset
│   ├── teacher_ft_roberta_wiki.py  # Teacher fine-tuning on Wiki dataset
├── evaluation/
│   ├── all_tests.py               # Comprehensive model testing
│   ├── error_analysis.py          # Model error analysis
│   ├── final_eval_script.py       # Final model evaluation
│   ├── gradient.py                # Gradient visualization
│   ├── old_model_test.py          # Legacy model testing
│   ├── old_model_test_csv.py      # CSV-based model testing
├── main.py                        # Main training execution script
├── config.yaml                    # Configuration parameters
├── Data/                          # Dataset directory
└── training_curve/                # Training visualizations
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd KTCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch 1.10+
- PyTorch Lightning 1.5+
- Transformers 4.0+
- Weights & Biases (wandb)
- Datasets (Hugging Face)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pyyaml

## Model Architecture

### Components

1. **Teacher Model**
   - RoBERTa-Large base
   - Fine-tuned on explicit hate speech data
   - Provides guidance for concept learning

2. **Student Model**
   - RoBERTa-Base architecture
   - Enhanced with concept refinement
   - Learns from teacher through knowledge transfer

3. **Autoencoder**
   - Maps between teacher and student spaces
   - Bottleneck dimension: 256
   - Facilitates knowledge transfer

4. **CAV Classifier**
   - Learns concept vectors
   - Uses prototype alignment
   - Guides concept refinement

## Training Pipeline

1. **Data Preparation**
```bash
# Configure dataset paths in config.yaml
# Ensure datasets are in correct format
```

2. **Teacher Model Training**
```bash
# Train on Wikipedia dataset
python training/teacher_ft_roberta_wiki.py

# Fine-tune on EA dataset
python training/teacher_ft_ea.py
```

3. **Student Model Training**
```bash
# Train student with concept refinement
python main.py
```

4. **Model Evaluation**
```bash
# Run comprehensive tests
python evaluation/all_tests.py

# Analyze errors
python evaluation/error_analysis.py

# Final evaluation
python evaluation/final_eval_script.py
```

## Configuration

Key parameters in `config.yaml`:

```yaml
aec:
  lightning_module:
    learning_rate: 0.005
    bottleneck_dim: 256
    max_epochs: 3

cav:
  lightning_module:
    learning_rate: 0.005
    max_epochs: 3

stmod:
  lightning_module:
    learning_rate: 0.00003
    lambda: 5
    n_clusters: 2

main:
  alpha: 0.3
  proto_update_frequency: 1
  cav_update_frequency: 2
  max_epochs: 3
```

## Datasets

The model is evaluated on three datasets:

1. **Wikipedia Toxicity (Wiki)**
   - General hate speech dataset
   - Used for initial training

2. **East Asian Prejudice (EA)**
   - Specific hate speech patterns
   - Used for concept refinement

3. **Covid-Hate (CH)**
   - Cross-domain evaluation
   - Tests generalization

## Evaluation

The project includes comprehensive evaluation tools:

1. **Performance Metrics**
   - F1 Score
   - ROC-AUC
   - Confusion Matrix
   - Gradient Analysis

2. **Visualization**
   - Training curves
   - Gradient heatmaps
   - Confusion matrices
   - Concept alignment plots

3. **Error Analysis**
   - False positive analysis
   - False negative analysis
   - Concept contribution analysis

## Logging and Monitoring

- Training metrics logged via WandB
- Loss curves saved in training_curve/
- Comprehensive error analysis in evaluation/

## Results

Performance metrics across datasets:

| Dataset | F1-Score | AUC |
|---------|----------|-----|
| Wiki    | 0.76     | 0.93|
| EA      | 0.69     | 0.91|
| CH      | 0.42     | 0.55|

## Citation

```bibtex
@misc{garg2024ktcrimprovingimplicithate,
      title={KTCR: Improving Implicit Hate Detection with Knowledge Transfer driven Concept Refinement}, 
      author={Samarth Garg and Vivek Hruday Kavuri and Gargi Shroff and Rahul Mishra},
      year={2024},
      eprint={2410.15314},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.15314}, 
}
```

## Acknowledgments

- Thanks to all contributors
- Built on RoBERTa architecture
- Uses PyTorch Lightning framework

## Disclaimer

This repository contains examples of explicit statements that are potentially offensive, used solely for research purposes in hate speech detection.