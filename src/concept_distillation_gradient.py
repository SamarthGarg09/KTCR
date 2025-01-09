import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams['figure.dpi'] = 300

# Load student model, tokenizer, CAV, and AEC
model_name = './models/student_roberta_base_wiki_3k_ea_balanced'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
student_model = RobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
student_model.eval()

# Assume CAV and AEC are preloaded models
cav_model = torch.load('../saved_models/concept_chcekpoints_wiki_plus_3k_student_5k_teacher_concept_all_balanced/cav.pth')
aec_model = torch.load('../saved_models/concept_chcekpoints_wiki_plus_3k_student_5k_teacher_concept_all_balanced/aec_model_mepoch_2.pth')

# Input sentence
sentence = "china isn't a race. it's a fucking country."

# Tokenize input
inputs = tokenizer(sentence, return_tensors='pt')

# Enable gradient calculation
for param in student_model.parameters():
    param.requires_grad = True

embedding_gradients = []

# Function to save gradients
def save_gradient(grad):
    embedding_gradients.append(grad)

# Hook for embedding gradients
embedding_layer = student_model.roberta.embeddings.word_embeddings
hook = embedding_layer.register_full_backward_hook(lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

# Forward pass through the student model
outputs = student_model(**inputs)
logits = outputs.logits

# Compute loss with CAV and AEC models
# Here you may need to adjust the loss function to reflect the combined effect of CAV, AEC, and student models
# This is a placeholder loss computation. Modify according to your model setup.
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = inputs['input_ids'][..., 1:].contiguous()

# Add the CAV and AEC effects (Assuming these return logits or loss values)
cav_logits = cav_model(inputs)
aec_logits = aec_model(inputs)

# Combine the loss from student model, CAV model, and AEC model
loss_fct = torch.nn.CrossEntropyLoss()
loss_student = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
loss_cav = loss_fct(cav_logits.view(-1, cav_logits.size(-1)), shift_labels.view(-1))  # Placeholder
loss_aec = loss_fct(aec_logits.view(-1, aec_logits.size(-1)), shift_labels.view(-1))  # Placeholder

# Total loss (This could be a weighted sum or another custom combination)
total_loss = loss_student + loss_cav + loss_aec

# Backward pass to calculate gradients
student_model.zero_grad()
total_loss.backward()

# Remove the hook after backward
hook.remove()

# Extract activations and gradients from the student model
last_hidden_state = outputs.hidden_states[-1][0]  
activations = last_hidden_state.detach().cpu().numpy()
grads = embedding_gradients[0][0].detach().cpu().numpy()

# Token processing
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
clean_tokens = [token.replace('Ġ', '') for token in tokens]

# Calculate norms for activations and gradients
activation_norms = np.linalg.norm(activations, axis=1)
gradient_norms = np.linalg.norm(grads, axis=1)

# Create DataFrame for easy plotting
df = pd.DataFrame({
    'Token': clean_tokens,
    'Activation Norm': activation_norms,
    'Gradient Norm': gradient_norms
})

# Function to plot the heatmaps
def plot_heatmaps(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 1]}, constrained_layout=True)

    sns.heatmap(
        [df['Activation Norm']],
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap='YlGnBu',
        cbar=True,
        xticklabels=df['Token'],
        yticklabels=['Activation Norm'],
        annot_kws={"fontsize":8}
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    axes[0].set_title('Activation Norms per Token', fontsize=12)

    sns.heatmap(
        [df['Gradient Norm']],
        ax=axes[1],
        annot=True,
        fmt=".2f",
        cmap='YlOrRd',
        cbar=True,
        xticklabels=df['Token'],
        yticklabels=['Gradient Norm'],
        annot_kws={"fontsize":8}
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    axes[1].set_title('Gradient Norms per Token', fontsize=12)

    plt.savefig('./gradients_wiki_ea_balanced_concept.png')

# Plot the heatmaps
plot_heatmaps(df)
