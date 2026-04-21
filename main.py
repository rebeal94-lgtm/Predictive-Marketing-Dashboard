import pandas as pd
df = pd.read_csv("ifood_df.csv")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nInfo:")
print (df.info())
print("\nDescribe;")
print(df.describe())
df = df.drop_duplicates()
print("\nValores nulos por columna:")
print(df.isnull().sum())
df = df.dropna()
print("\nNuevo shape después de la limpieza:", df.shape)
df['Total_spending'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)
df['Total_Purchases'] = (
    df['NumWebPurchases'] +
    df['NumCatalogPurchases'] +
    df['NumStorePurchases']
)
df['Total_Campaigns'] = (
    df['AcceptedCmp1'] +
    df['AcceptedCmp2'] +
    df['AcceptedCmp3'] +
    df['AcceptedCmp4'] +
    df['AcceptedCmp5']
)
df['Conversion_Rate'] = df['Total_Campaigns'] / 5
df['Engagement'] = df['Total_Purchases'] / (df['Recency'] + 1)
print("\nNuevas Columnas creadas")
print(df[['Total_spending' , 'Total_Purchases' , 'Total_Campaigns' , 'Conversion_Rate' , 'Engagement']].head())
X = df[['Income','Total_spending', 'Total_Purchases','Recency']]
y = df['Response']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state= 42
)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
df['Prediction'] = model.predict(X)
print("\nPredicciones:")
print(df[['Response', 'Prediction']].head())
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("\nAccuracy del modelo:", accuracy_score(y_test, y_pred))
df.to_csv("clean_marketing_data.csv", index=False)
print("\nArchivo exportado correctamente")