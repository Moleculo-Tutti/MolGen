import pandas as pd
import torch

from tqdm import tqdm
from path import Path


from sklearn.model_selection import train_test_split
from DataPipeline.preprocessing import process_encode_graph

csv_path = Path(".\data") / "rndm_zinc_drugs_clean_3.csv"

zinc_df = pd.read_csv(csv_path)

def remove_iodine_bromine_phosphorus(df):

    df = df[df['smiles'].str.contains('Br') == False]
    df = df[df['smiles'].str.contains('I') == False]
    df = df[df['smiles'].str.contains('P') == False]
    
    return df


def main():
    preprocessed_graph = []
    filtered_df = remove_iodine_bromine_phosphorus(zinc_df)

    for row in tqdm(filtered_df.itertuples()):
        smiles = row.smiles
        data = process_encode_graph(smiles, 'reduced')
        preprocessed_graph.append(data)

    # Separate data into train, validation and test sets
    X_train_val, X_test = train_test_split(preprocessed_graph, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1111, random_state=42)

    # Save data sets into files
    torch.save(X_train, 'data/preprocessed_graph_train_no_I_Br_P.pt')
    torch.save(X_val, 'data/preprocessed_graph_val_no_I_Br_P.pt')
    torch.save(X_test, 'data/preprocessed_graph_test_no_I_Br_P.pt')

if __name__ == "__main__":
    main()