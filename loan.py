#import modules yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#load dataset
df = pd.read_csv("loan.csv")
print(df.shape)
df.head(5)
#meihat kolom yang ada
df.columns

#membedakan numerik kolom dengan kategorik kolom
numcols = ['loan_amnt', 'int_rate', 'installment']
catcols = ['term', 'grade', 'sub_grade']


#data baru(menggabungkan numerik dan kategorik kolom)
data = df[numcols+catcols]


#melihat term distribusi
sns.set(rc={'figure.figsize':(7,7)})
sns.countplot(x="term", data=data)

#melihat loan_amnt distribusi
data["loan_amnt"].plot.hist()

#tabel korelasi untuk melihat feature mana yg berkorelasi tinggi untuk dibuat prediksi
correlation = data.corr()
plt.figure(figsize = (10, 10))
sns.heatmap(correlation, vmax = 1, square = True, annot = True)

#deskripsi data
data.describe()

#menentukan X dan Y untuk diprediksi
#disini saya memprediksi loan_amnt menggunakan feature installment karena didapat hasil yang tinggi di tabel korelasi
#saya membedakan loan_amount menjadi 2 yaitu tinggi dan rendah
Y = data['loan_amnt']>=1.290000e+04
X = data.copy()
del X['loan_amnt']
len(X.columns)


#menghilangkan NaN
data = data.dropna()

#Split-out validation dataset
from sklearn.model_selection import train_test_split

X = data.iloc[:,2:3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 10, random_state = seed)
kfold

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

cv_logistic_reg = cross_val_score(LogisticRegression(),
                                 X_train, Y_train, cv=kfold, scoring = 'accuracy')
cv_logistic_reg

#Spot-Check Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = []
models.append(( 'LR' , LogisticRegression()))
models.append(( 'LDA' , LinearDiscriminantAnalysis()))
models.append(( 'KNN' , KNeighborsClassifier()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=10, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring= 'accuracy' )
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
  
#Membuat predictions di validation dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#saya menggunakan KNN untuk prediksi karena akurasinya adalah yang paling tinggi
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

hasil = X_validation
hasil["label_predict"] = predictions
hasil["actual_label"] = Y_validation
hasil[hasil["label_predict"] != hasil["actual_label"]]

#Hasil prediksi menggunakan KNN
hasil.head(5)
