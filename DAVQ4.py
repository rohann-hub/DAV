import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import accuracy_score
df = pd.DataFrame([
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'bread', 'eggs'],
    ['bread']
], columns=['item1', 'item2', 'item3'])
encoded = df.apply(lambda row: pd.Series(row.dropna().values), axis=1).stack().reset_index(level=1, drop=True).to_frame('item')
encoded['txn'] = encoded.index
basket = pd.crosstab(encoded['txn'], encoded['item']).astype(bool).astype(int)
freq_items = apriori(basket, min_support=0.5, use_colnames=True)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)
preds = rules['consequents'].apply(lambda x: list(x)[0] in basket.columns)
truths = rules['antecedents'].apply(lambda x: all(i in basket.columns for i in x))
print("Accuracy:", accuracy_score(truths, preds))
