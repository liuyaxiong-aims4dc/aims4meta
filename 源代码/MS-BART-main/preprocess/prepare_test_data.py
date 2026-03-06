import pandas as pd
import os

def prepare_test_data_massspecgym(file_path, fold="test", dataset_name="MassSpecGym"):
    df = pd.read_csv(file_path, sep='\t')
    fold_key = "fold" if dataset_name == "MassSpecGym"  else "split"
    test_df = df[df[fold_key] == fold]
    
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    output_dir = os.path.join(file_dir, fold)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)

    test_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    dataset_name = "MassSpecGymL" # MassSpecGym/CANOPUS
    prepare_test_data_massspecgym(f"logs/datasets/{dataset_name}/{dataset_name}_fps_selfies_threshold_0.2.tsv", "train", dataset_name)
    prepare_test_data_massspecgym(f"logs/datasets/{dataset_name}/{dataset_name}_fps_selfies_threshold_0.2.tsv", "test", dataset_name)
    prepare_test_data_massspecgym(f"logs/datasets/{dataset_name}/{dataset_name}_fps_selfies_threshold_0.2.tsv", "val", dataset_name)
