from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def audMetrics(output, labels, preds):
    
    metric_dict = {"accuracy" : None, 
                   "precision" : None, 
                   "recall" : None, 
                   "f1" : None 
    }

    metric_dict["accuracy"] = accuracy_score(labels, preds )
    metric_dict["precision"] = precision_score(labels, preds, average = 'micro' )
    metric_dict["recall"] = recall_score(labels, preds, average = 'micro' )
    metric_dict["f1"] = f1_score(labels, preds, average = 'micro' )

    return metric_dict

