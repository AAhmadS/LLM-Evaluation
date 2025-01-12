import pandas as pd
import os
import pickle

def load_data_info(info_dir, data_name):
    loading_dir = os.path.join(info_dir, data_name)
    infos = {"dataset_name":data_name}
    files_names = os.listdir(loading_dir)
    files_dict = {f"{name.split('.pkl')[0]}": pickle.load(open(os.path.join(loading_dir, f"{name}"), 'rb')) for name in files_names if name.endswith(".pkl")}
    infos.update(files_dict)
    return infos

class Dataset:
    def __init__(self, data_dir, info_dir, data_name):
        self.load_data(data_dir, data_name)
        self.load_info(info_dir, data_name)

        
    def load_data(self, data_dir, data_name):
        self.data = pd.read_csv(os.path.join(data_dir,data_name)+".csv")
        self.selected_subset = pd.read_csv(os.path.join(data_dir,data_name)+"_test.csv")
        
    def load_info(self, info_dir, data_name):
        information = load_data_info(info_dir, data_name)
        self.task_description = information["task_description"]
        self.dataset_description = information["dataset_description"]
        self.overall_score_column_name = information["overall_score_column_name"]
        self.content_column_name = information["content_column_name"]
