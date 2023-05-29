import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re




def plot_history_GNN1(csv_file, legend_dict=None):
    df = pd.read_csv(csv_file)
    conversion = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'S', 5: 'Cl', 6:'stop'}
    # Conversion des colonnes de type string en listes
    df['avg_output_vector'] = df['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df['avg_label_vector'] = df['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df['precision'] = df['precision'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    df['recall'] = df['recall'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    # Graphique pour la loss
    axs[0, 0].plot(df['epoch'], df['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')

    axs[0, 1].plot(df['epoch'], np.log(df['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()

    # Graphique pour avg_output_vector / avg_label_vector
    for i in range(len(df['avg_output_vector'][0])):
        ratio = df['avg_output_vector'].apply(lambda x: x[i]) / df['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df['epoch'], ratio, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1, 0].legend()

    # Graphique pour la precision
    for i in range(len(df['precision'][0])):
        precision =  df['precision'].apply(lambda x: x[i])
        axs[1, 1].plot(df['epoch'], precision, label=conversion[i])

    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision by epochs')
    axs[1, 1].legend()

    # Graphique pour le recall
    for i in range(len(df['recall'][0])):
        recall = df['recall'].apply(lambda x: x[i])
        axs[2, 0].plot(df['epoch'], recall, label=conversion[i])

    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall by epochs')
    axs[2, 0].legend()

    for i in range(len(df['recall'][0])):
        recall = df['recall'].apply(lambda x: x[i])
        precision =  df['precision'].apply(lambda x: x[i])
        axs[2, 1].plot(df['epoch'], 2/(1/recall + 1/precision), label=conversion[i])

    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('F1_score by epochs')
    axs[2, 1].legend()

    axs[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[2, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[2, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Ajustement de l'espacement entre les sous-graphiques
    fig.tight_layout()

    # Affichage de la figure
    plt.show()


def plot_history_GNN2(csv_file, legend_dict=None):
    df = pd.read_csv(csv_file)
    conversion = {0: 'aromatic', 1: 'single', 2: 'double', 3: 'triple'}
    # Conversion des colonnes de type string en listes
    df['avg_output_vector'] = df['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df['avg_label_vector'] = df['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df['precision'] = df['precision'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    df['recall'] = df['recall'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    # Graphique pour la loss
    axs[0, 0].plot(df['epoch'], df['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')

    axs[0, 1].plot(df['epoch'], np.log(df['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()

    # Graphique pour avg_output_vector / avg_label_vector
    for i in range(len(df['avg_output_vector'][0])):
        ratio = df['avg_output_vector'].apply(lambda x: x[i]) / df['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df['epoch'], ratio, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1, 0].legend()

    # Graphique pour la precision
    for i in range(len(df['precision'][0])):
        precision =  df['precision'].apply(lambda x: x[i])
        axs[1, 1].plot(df['epoch'], precision, label=conversion[i])

    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision by epochs')
    axs[1, 1].legend()

    # Graphique pour le recall
    for i in range(len(df['recall'][0])):
        recall = df['recall'].apply(lambda x: x[i])
        axs[2, 0].plot(df['epoch'], recall, label=conversion[i])

    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall by epochs')
    axs[2, 0].legend()

    for i in range(len(df['recall'][0])):
        recall = df['recall'].apply(lambda x: x[i])
        precision =  df['precision'].apply(lambda x: x[i])
        axs[2, 1].plot(df['epoch'], 2/(1/recall + 1/precision), label=conversion[i])

    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('F1_score by epochs')
    axs[2, 1].legend()

    axs[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[2, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axs[2, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Ajustement de l'espacement entre les sous-graphiques
    fig.tight_layout()

    # Affichage de la figure
    plt.show()



def plot_history_GNN3(csv_file, legend_dict=None):
    df = pd.read_csv(csv_file)
    conversion = {0: 'Aromatic close', 1: 'Simple close', 2: 'Double closing', 3: 'Triple closing', 4: 'No closing'}
    # Conversion des colonnes de type string en listes
    df['avg_output_vector'] = df['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df['avg_label_vector'] = df['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(5, 2, figsize=(10, 8))

    # Graphique pour la loss
    axs[0, 0].plot(df['epoch'], df['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')

    axs[0, 1].plot(df['epoch'], np.log(df['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()

    # Graphique pour avg_output_vector / avg_label_vector
    for i in range(len(df['avg_output_vector'][0])):
        ratio = df['avg_output_vector'].apply(lambda x: x[i]) / df['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df['epoch'], ratio, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1, 0].legend()

    axs[1, 1].plot(df['epoch'], df['pseudo_precision'])
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision (created a cycle at good time / all cycles created)')
    axs[1, 1].legend()

    axs[2, 0].plot(df['epoch'], df['pseudo_recall'])
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall (Cycles created at good time / all cycles should have created)')
    axs[2, 0].legend()

    axs[2, 1].plot(df['epoch'], df['pseudo_recall_placed'])
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('Recall (Cycles created at good place / all cycles should have created)')
    axs[2, 1].legend()

    axs[3, 0].plot(df['epoch'], df['pseudo_recall_type'])
    axs[3, 0].set_xlabel('Epoch')
    axs[3, 0].set_ylabel('Value')
    axs[3, 0].set_title('Recall (Cycles created at good place with good bound / all cycles should have created)')
    axs[3, 0].legend()

    axs[3, 1].plot(df['epoch'], df['f1_score'])
    axs[3, 1].set_xlabel('Epoch')
    axs[3, 1].set_ylabel('Value')
    axs[3, 1].set_title('F1_score')
    axs[3, 1].legend()

    axs[4, 0].plot(df['epoch'], df['conditionnal_precision_placed'])
    axs[4, 0].set_xlabel('Epoch')
    axs[4, 0].set_ylabel('Value')
    axs[4, 0].set_title('Precision (Cycles created at good place / all cycles created at good time)')
    axs[4, 0].legend()
    
    axs[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


    # Ajustement de l'espacement entre les sous-graphiques
    fig.tight_layout()

    # Affichage de la figure
    plt.show()