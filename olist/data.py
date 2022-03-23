import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # Hints 1: Build csv_path as "absolute path" in order to call this method from anywhere.
            # Do not hardcode your path as it only works on your machine ('Users/username/code...')
            # Use __file__ instead as an absolute path anchor independant of your usename
            # Make extensive use of `breakpoint()` to investigate what `__file__` variable is really
        # Hint 2: Use os.path library to construct path independent of Mac vs. Unix vs. Windows specificities
        # csv_path
        project_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        # alternative:project_path = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(project_path, 'data', 'csv')
        ## list file_names, but only .csv (parse strings)
        file_names = [file for file in os.listdir(csv_path) if file[-3:] == 'csv']
        # alternative: file.endswith = '.csv'
        ## list key_names
        key_names = [key.rstrip('.csv').lstrip('olist').rstrip('dataset').rstrip('_').lstrip('_') for key in file_names]
        ## make dictionary
        data = {}
        for key, filename in zip(key_names, file_names):
            data[key] = pd.read_csv(os.path.join(csv_path, filename))
        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
