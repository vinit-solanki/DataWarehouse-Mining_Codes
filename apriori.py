import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules
data = {
    'Bread': [1, 1, 1, 0, 0],
    'Milk': [1, 1, 0, 1, 1],
    'Cheese': [1, 0, 1, 1, 0],
    'Butter': [0, 1, 1, 0, 0],
    'Eggs': [0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)
# Step-1: Apply Apriori Algorithm and find freq itemset to Generate Association Rules
freqt_itemset = apriori(df,min_support=0.2,use_colnames=True)
print("Frequent Itemset:", freqt_itemset.head())
# Step-2 : Generate association rules
rules = association_rules(freqt_itemset,metric="confidence",min_threshold=0.5)
print("Association Rules: ",rules.head())