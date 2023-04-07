import pandas as pd
from model.svm_model import SVMModel
import csv
import joblib

path_file_input = 'data_assistants.csv'
path_file_output = 'model.pkl'

def train_data():
    train_data = []
    with open(path_file_input,'rt', encoding='utf8')as f:
        data_assistant = csv.reader(f)
        for row in data_assistant:
            if(row[0]=="" and row[1]==""):
                pass
            else:
                #training data form file data_assistant.csv
                train_data.append({"feature": u"{0}".format(row[0]), "target": u"{0}".format(row[1])}) 
    df_train_data = pd.DataFrame(train_data)
    
    model = SVMModel() 
    model.fit(df_train_data["feature"], df_train_data["target"])
    joblib.dump(model, path_file_output) 
    
def detect_content(text, intent=""):
    test_data = [{"feature": f"{text}", "target": "{intent}"}]
    df_test = pd.DataFrame(test_data)
    
    clf = joblib.load(path_file_output)
    resp = clf.predict(df_test["feature"])
    return resp[0] if resp else ""
