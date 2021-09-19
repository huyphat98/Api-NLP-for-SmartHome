#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from model.naive_bayes_model import NaiveBayesModel
from model.svm_model import SVMModel
import csv, json, os
path_cfg = os.getcwd()
config_file = "{}/config.json".format(path_cfg, )
config = json.loads(open(config_file, 'r').read())

class TextClassificationPredict(object):
    def __init__(self):
        #train data
        train_data = []
        with open(config['path']['traningFile'],'rt', encoding='utf8')as f:
          data_assistant = csv.reader(f)
          for row in data_assistant:
                if(row[0]=="" and row[1]==""):
                    pass
                else:
                    train_data.append({"feature": u"{0}".format(row[0]), "target": u"{0}".format(row[1])}) #training data form file data_assistant.csv

        global df_train
        df_train = pd.DataFrame(train_data)

    def classification(self, text, intent=""):
        self.text = text
        self.intent = intent
        test_data = []
        test_data.append({"feature": f"{self.text}", "target": "{self.intent}"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = NaiveBayesModel()
        clf = model.clf.fit(df_train["feature"], df_train.target)
        predicted = clf.predict(df_test["feature"])
        # print(predicted,"\n")
        # print (clf.predict_proba(df_test["feature"]),"\n")
        return predicted

if __name__ == '__main__':
    tcp = TextClassificationPredict() # khởi tạo object
    data = tcp.classification(text = str(input("Nhập text: ")))
    print(type(data))
    print(str(data[0]))