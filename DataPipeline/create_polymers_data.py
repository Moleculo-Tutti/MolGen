import pandas as pd
import torch

from tqdm import tqdm
from path import Path

from rdkit import Chem

from sklearn.model_selection import train_test_split
from preprocessing import process_encode_graph

txt_path = Path("./data") / "all_polymers.txt"

df_polymers = pd.read_csv(txt_path)



def main():
    preprocessed_graph = []
    
    smiles_list = df_polymers.iloc[:,0].to_list()
    for smiles in tqdm(smiles_list):
        data = process_encode_graph(smiles, 'polymer', kekulize=True)
        preprocessed_graph.append(data)

    # Separate data into train, validation and test sets
    X_train_val, X_test = train_test_split(preprocessed_graph, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1111, random_state=42)

    # Save data sets into files
    torch.save(X_train, 'data/preprocessed_graph_train_polymers_kekulized.pt')
    torch.save(X_val, 'data/preprocessed_graph_val_polymers_kekulized.pt')
    torch.save(X_test, 'data/preprocessed_graph_test_polymers_kekulized.pt')

if __name__ == "__main__":
    main()