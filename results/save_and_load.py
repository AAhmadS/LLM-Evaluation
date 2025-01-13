import os

base_path = os.path.dirname(os.path.abspath(__file__))
def save_results(phase,
                 results, 
                 dataset_name,
                 setting_name
                 ):
        """
        Save the results to a JSON file.
        """
        save_path = os.path.join(base_path, phase, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        results.to_csv(f"{os.path.join(save_path, setting_name)}.csv")