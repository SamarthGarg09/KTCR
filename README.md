# KTCR: Knowledge Tranfer driven Concept Refinement

## File Structure:

`Data:` This folder contains the data used and also conatin a folder called Augmentation in it, which is used for augmenting data based on the DoE score

`src:` This folder contains all our source code. `main.py` runs the KTCR framework and saves the models. Before running it, train the student and teacher models using the scripts in the same folder. (Ensure you change the paths to datasets as you require). All the config is available in `config.yaml`

`training_curve:` This folder contains the training curves of different models used in the KTCR method and also contains the plot of Concept Loss.

