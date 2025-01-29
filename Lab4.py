from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample dataset
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter']
]

# Convert the dataset into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("One-hot encoded Data:")
print(df)

# Find frequent itemsets with a minimum support of 0.2
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate strong association rules with a minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nStrong Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
