import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re




def plot_history_GNN1(csv_file, charged=False):
    df = pd.read_csv(csv_file)
    if charged:
        conversion = {0: 'C', 1: 'N', 2: 'N+', 3: 'N-', 4:'O', 5:'O-', 6:'F', 7:'S', 8:'S-', 9:'Cl', 10:'Br', 11:'I', 12:'Stop'}
    else:
        conversion = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'S', 5:'Cl', 6:'Br', 7:'I'}

    # Creating a df for each metric (loss, avg_output_vector, avg_label_vector, precision, recall)

    df['epoch'] = df['epoch'].astype(int)

    df_loss = df[['epoch', 'loss']]
    df_loss = df_loss.dropna()

    df_avg_output_vector = df[['epoch', 'avg_output_vector']]
    df_avg_output_vector = df_avg_output_vector.dropna()

    df_avg_label_vector = df[['epoch', 'avg_label_vector']]
    df_avg_label_vector = df_avg_label_vector.dropna()

    df_precision = df[['epoch', 'precision']]
    df_precision = df_precision.dropna()

    df_recall = df[['epoch', 'recall']]
    df_recall = df_recall.dropna()

    # Conversion des colonnes de type string en listes
    df_avg_output_vector['avg_output_vector'] = df_avg_output_vector['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df_avg_label_vector['avg_label_vector'] = df_avg_label_vector['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df_precision['precision'] = df_precision['precision'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    df_recall['recall'] = df_recall['recall'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    # Graphique pour la loss
    axs[0, 0].plot(df_loss['epoch'], df_loss['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')
    axs[0, 0].legend()

    axs[0, 1].plot(df_loss['epoch'], np.log(df_loss['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()
    
    # Graphique pour avg_output_vector / avg_label_vector

    for i in range(len(df_avg_output_vector['avg_output_vector'][0])):
        ratio = df_avg_output_vector['avg_output_vector'].apply(lambda x: x[i]) / df_avg_label_vector['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df_avg_output_vector['epoch'], ratio.values, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1,0].legend()

    df_precision['precision_length'] = df_precision['precision'].apply(len)

    # Delete rows with precision_length not equal to 13
    df_precision = df_precision[df_precision['precision_length'] == len(conversion)]

    # Remove the precision_length column
    df_precision = df_precision.drop('precision_length', axis=1)


    # Graphique pour la precision
    for i in range(len(df_precision['precision'][0])):
        precision =  df_precision['precision'].apply(lambda x: x[i])
        axs[1, 1].plot(df_precision['epoch'], precision, label=conversion[i])
    print("stop is not the bug")

    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision by epochs')

    # Graphique pour le recall
    for i in range(len(df_recall['recall'][0])):
        recall = df_recall['recall'].apply(lambda x: x[i])
        axs[2, 0].plot( df_recall['epoch'], recall, label=conversion[i])

    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall by epochs')

    for i in range(len(df_recall['recall'][0])):
        recall = df_recall['recall'].apply(lambda x: x[i])
        precision =  df_precision['precision'].apply(lambda x: x[i] )
        f1_score = 2/(1/recall + 1/precision)
        axs[2, 1].plot(df_recall['epoch'], f1_score, label=conversion[i])

    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('F1_score by epochs')

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
    
    df['epoch'] = df['epoch'].astype(int)

    df_loss = df[['epoch', 'loss']]
    df_loss = df_loss.dropna()

    df_avg_output_vector = df[['epoch', 'avg_output_vector']]
    df_avg_output_vector = df_avg_output_vector.dropna()

    df_avg_label_vector = df[['epoch', 'avg_label_vector']]
    df_avg_label_vector = df_avg_label_vector.dropna()

    df_precision = df[['epoch', 'precision']]
    df_precision = df_precision.dropna()

    df_recall = df[['epoch', 'recall']]
    df_recall = df_recall.dropna()

    # Conversion des colonnes de type string en listes
    df_avg_output_vector['avg_output_vector'] = df_avg_output_vector['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df_avg_label_vector['avg_label_vector'] = df_avg_label_vector['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df_precision['precision'] = df_precision['precision'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    df_recall['recall'] = df_recall['recall'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x) if num != 'tensor'])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    # Graphique pour la loss
    axs[0, 0].plot(df_loss['epoch'], df_loss['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')
    axs[0, 0].legend()

    axs[0, 1].plot(df_loss['epoch'], np.log(df_loss['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()
    
    # Graphique pour avg_output_vector / avg_label_vector

    for i in range(len(df_avg_output_vector['avg_output_vector'][0])):
        ratio = df_avg_output_vector['avg_output_vector'].apply(lambda x: x[i]) / df_avg_label_vector['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df_avg_output_vector['epoch'], ratio.values, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1,0].legend()

    df_precision['precision_length'] = df_precision['precision'].apply(len)

    # Delete rows with precision_length not equal to 13
    df_precision = df_precision[df_precision['precision_length'] == len(conversion)]

    # Remove the precision_length column
    df_precision = df_precision.drop('precision_length', axis=1)


    # Graphique pour la precision
    for i in range(len(df_precision['precision'][0])):
        precision =  df_precision['precision'].apply(lambda x: x[i])
        axs[1, 1].plot(df_precision['epoch'], precision, label=conversion[i])
    print("stop is not the bug")

    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision by epochs')

    # Graphique pour le recall
    for i in range(len(df_recall['recall'][0])):
        recall = df_recall['recall'].apply(lambda x: x[i])
        axs[2, 0].plot( df_recall['epoch'], recall, label=conversion[i])

    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall by epochs')

    for i in range(len(df_recall['recall'][0])):
        recall = df_recall['recall'].apply(lambda x: x[i])
        precision =  df_precision['precision'].apply(lambda x: x[i] )
        f1_score = 2/(1/recall + 1/precision)
        axs[2, 1].plot(df_recall['epoch'], f1_score, label=conversion[i])

    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('F1_score by epochs')

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
    conversion = {0: 'Aromatic close', 1: 'Simple close', 2: 'Double closing', 3: 'No closing'}
    # Conversion des colonnes de type string en listes

    df['epoch'] = df['epoch'].astype(int)

    df_loss = df[['epoch', 'loss']]
    df_loss = df_loss.dropna()

    df_avg_output_vector = df[['epoch', 'avg_output_vector']]
    df_avg_output_vector = df_avg_output_vector.dropna()

    df_avg_label_vector = df[['epoch', 'avg_label_vector']]
    df_avg_label_vector = df_avg_label_vector.dropna()

    df_pseudo_precision = df[['epoch', 'pseudo_precision']]
    df_pseudo_precision = df_pseudo_precision.dropna()

    df_pseudo_recall = df[['epoch', 'pseudo_recall']]
    df_pseudo_recall= df_pseudo_recall.dropna()

    df_pseudo_recall_placed = df[['epoch', 'pseudo_recall_placed']]
    df_pseudo_recall_placed = df_pseudo_recall_placed.dropna()

    df_conditionnal_precision_placed = df[['epoch', 'conditionnal_precision_placed']]
    df_conditionnal_precision_placed = df_conditionnal_precision_placed.dropna()

    df_pseudo_recall_type = df[['epoch', 'pseudo_recall_type']]
    df_pseudo_recall_type = df_pseudo_recall_type.dropna()

    df_f1_score = df[['epoch', 'f1_score']]
    df_f1_score = df_f1_score.dropna()

    df_avg_output_vector['avg_output_vector'] = df_avg_output_vector['avg_output_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    df_avg_label_vector['avg_label_vector'] = df_avg_label_vector['avg_label_vector'].apply(lambda x: [float(num) for num in re.findall(r'\d+\.\d+', x)])
    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(5, 2, figsize=(15, 10))

    # Graphique pour la loss
    axs[0, 0].plot(df_loss['epoch'], df_loss['loss'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss by epochs')

    axs[0, 1].plot(df_loss['epoch'], np.log(df_loss['loss']))
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Log_Loss')
    axs[0, 1].set_title('Log_Loss by epochs')
    axs[0, 1].legend()

    # Graphique pour avg_output_vector / avg_label_vector
    for i in range(len(conversion)):
        ratio = df_avg_output_vector['avg_output_vector'].apply(lambda x: x[i]) / df_avg_label_vector['avg_label_vector'].apply(lambda x: x[i])
        axs[1, 0].plot(df_avg_output_vector['epoch'], ratio, label=conversion[i])

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('avg_output_vector / avg_label_vector by epochs')
    axs[1, 0].legend()

    axs[1, 1].plot(df_pseudo_precision['epoch'], df_pseudo_precision['pseudo_precision'])
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Precision (created a cycle at good time / all cycles created)')
    axs[1, 1].legend()

    axs[2, 0].plot(df_pseudo_recall['epoch'], df_pseudo_recall['pseudo_recall'])
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].set_title('Recall (Cycles created at good time / all cycles should have created)')
    axs[2, 0].legend()

    axs[2, 1].plot(df_pseudo_recall_placed['epoch'], df_pseudo_recall_placed['pseudo_recall_placed'])
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('Recall (Cycles created at good place / all cycles should have created)')
    axs[2, 1].legend()

    axs[3, 0].plot(df_pseudo_recall_type['epoch'], df_pseudo_recall_type['pseudo_recall_type'])
    axs[3, 0].set_xlabel('Epoch')
    axs[3, 0].set_ylabel('Value')
    axs[3, 0].set_title('Recall (Cycles created at good place with good bound / all cycles should have created)')
    axs[3, 0].legend()

    axs[3, 1].plot(df_f1_score['epoch'], df_f1_score['f1_score'])
    axs[3, 1].set_xlabel('Epoch')
    axs[3, 1].set_ylabel('Value')
    axs[3, 1].set_title('F1_score')
    axs[3, 1].legend()

    axs[4, 0].plot(df_conditionnal_precision_placed['epoch'], df_conditionnal_precision_placed['conditionnal_precision_placed'])
    axs[4, 0].set_xlabel('Epoch')
    axs[4, 0].set_ylabel('Value')
    axs[4, 0].set_title('Precision (Cycles created at good place / all cycles created at good time)')
    axs[4, 0].legend()
    
    axs[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


    # Ajustement de l'espacement entre les sous-graphiques
    fig.tight_layout()

    # Affichage de la figure
    plt.show()