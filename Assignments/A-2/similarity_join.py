# similarity_join.py
import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)
          
    def preprocess_df(self, df, cols): 
        """
            Write your code!
        """
        #perform the preprocessing on the required two columns separately and then combine them
        df[cols[0]] = df[cols[0]].fillna('')
        df[cols[1]] = df[cols[1]].fillna('')
        
        col0 = df[cols[0]].str.lower()
        col1 = df[cols[1]].str.lower()
        
        col0 = df[cols[0]].str.split(r'\W+')
        col1 = df[cols[1]].str.split(r'\W+')
        #eliminating blank values in the joinKey lists
        col0 = col0.apply(lambda x: [i for i in x if i])
        col1 = col1.apply(lambda x: [i for i in x if i])
        df["joinKey"] = col0 + col1
        return df
    
    def filtering(self, df1, df2):
        """
            Write your code!
        """
        #flattening the joinKey lists
        open_df1 = df1.explode("joinKey")
        open_df2 = df2.explode("joinKey")
        #finding common joinKey values between the two dataframes
        inter_df = open_df1.merge(open_df2, on=["joinKey"])
        inter_df = inter_df.drop_duplicates(subset=["id_x","id_y"]) 
        can_df = pd.DataFrame({ "id1":inter_df["id_x"],
                               "joinKey1":inter_df["id_x"].map(df1.set_index("id")["joinKey"].astype(str)),
                               "id2":inter_df["id_y"],
                               "joinKey2":inter_df["id_y"].map(df2.set_index("id")["joinKey"].astype(str))
                               })
        return can_df
    
    def jaccard_value(self,x,y):
      x = x.strip('][').split(', ')
      y = y.strip('][').split(', ')
      x = set(x)
      y = set(y)
      jaccard = float(len(x.intersection(y))) / len(x.union(y))
      return jaccard

    def verification(self, cand_df, threshold):
        """
            Write your code!
        """
        result_df = cand_df.copy()
        #get the jaccard value for the two joinKeys
        result_df["jaccard"] = cand_df.apply(lambda x: self.jaccard_value(x["joinKey1"], x["joinKey2"]), axis=1)
        result_df = result_df[result_df["jaccard"]>threshold]
        return result_df
        
    def evaluate(self, result, ground_truth):
        """
            Write your code!
        """
        intersection = [[x for x in sublist if x in ground_truth] for sublist in result]
        pression = len(intersection) / len(result)
        recall = len(intersection) / len(ground_truth)
        fmeasure = 2*pression*recall / (pression+recall)
        return (pression, recall, fmeasure)
        
    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))
        return result_df
       

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)
    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))

