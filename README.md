# k-NearestNeighbors

In this project, we will use k-nearest neighbors classifier to analyze the conditions that influence the income (<=50K or >50K)

## 1. Drop the unknown value to Nan
```python
df = df.replace('?', np.NaN)
df.isnull().sum()
df = df.dropna(axis=0)
df.reset_index(drop=True, inplace=True)
df.head()
```

## 2. Normalize the dependent variable
```python
df['income'].value_counts(normalize=True)
```

## 3. Split the variables into str and int
```python
df_str = df.select_dtypes(include='object')
df_int = df.select_dtypes(exclude='object')
```

## 4. Convert the values of str variables to binary
```python
df_str = pd.get_dummies(df_str)
```

## 5. Check the correlation between int variables
```python
plt.figure(figsize=(16,6))
mask = np.triu(np.ones_like(df_int.corr(),dtype=bool))
heatmap = sns.heatmap(df_int.corr(), mask=mask, vmin=-1,vmax=1,annot=True,cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
```
![image](https://github.com/ThomasWongHY/k-NearestNeighbors/assets/86035047/85bb10f4-d512-4d32-8207-c36f2379c602)

## 6. Visualize the relationship between int variables and dependent variable (income)
### Histogram
```python
for col in df_int.columns.to_list():
    if col != 'income':
        sns.displot(x=col, hue='income', data=df_int, kde=True)
```
![image](https://github.com/ThomasWongHY/k-NearestNeighbors/assets/86035047/399af91a-95ea-4028-b6eb-b928ce21cdfb)
![image](https://github.com/ThomasWongHY/k-NearestNeighbors/assets/86035047/65e7875a-90a9-4397-a039-0d69d6049599)
![image](https://github.com/ThomasWongHY/k-NearestNeighbors/assets/86035047/636b5000-a99e-46f1-8e91-38ddf98ae638)

### Box Plot
```python
for col in df_int.columns.to_list():
    if col != 'income':
        sns.catplot(x='income', y=col, kind='box', data=df)
```

## 7. Preprocess the int data by Standard Scaler
```python
ss = StandardScaler()
X = ss.fit_transform(X)
```

## 8. Train the model
```python
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Create the k-nearest neighbors classifier and fit the data
knn = KNeighborsClassifier().fit(X_train, y_train)

# Predict the income by testing data
y_pred = knn.predict(X_test)

# Evaluate the results
accuracy_score(y_test, y_pred)
```

## 9. Check the error rate by graph
```python
error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(20,10))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```
![image](https://github.com/ThomasWongHY/k-NearestNeighbors/assets/86035047/3763fd49-8b61-4b97-a92a-7ad5a7e2c6e0)

## 10. Retrain with new K value
```python
knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

accuracy_score(y_test, pred)
print(classification_report(y_test,pred))
```
