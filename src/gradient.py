import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['figure.dpi'] = 300

model_name = './models/student_roberta_base_wiki_3k_ea_balanced'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(
    model_name, output_hidden_states=True
)
model.eval()

sentence = "china isn't a race. it's a fucking country."

inputs = tokenizer(sentence, return_tensors='pt')

for param in model.parameters():
    param.requires_grad = True

embedding_gradients = []

def save_gradient(grad):
    embedding_gradients.append(grad)

embedding_layer = model.roberta.embeddings.word_embeddings

hook = embedding_layer.register_full_backward_hook(
    lambda module, grad_input, grad_output: save_gradient(grad_output[0])
)

outputs = model(**inputs)
logits = outputs.logits

shift_logits = logits[..., :-1, :].contiguous()
shift_labels = inputs['input_ids'][..., 1:].contiguous()
loss_fct = torch.nn.CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
)

model.zero_grad()
loss.backward()

hook.remove()

last_hidden_state = outputs.hidden_states[-1][0]  
activations = last_hidden_state.detach().cpu().numpy()

grads = embedding_gradients[0][0].detach().cpu().numpy()  

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

clean_tokens = [token.replace('Ġ', '') for token in tokens]

activation_norms = np.linalg.norm(activations, axis=1)
gradient_norms = np.linalg.norm(grads, axis=1)

import pandas as pd
df = pd.DataFrame({
    'Token': clean_tokens,
    'Activation Norm': activation_norms,
    'Gradient Norm': gradient_norms
})

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


    plt.savefig('./gradients_wiki_ea_balanced.png')

plot_heatmaps(df)