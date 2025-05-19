#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns 


# In[4]:


df=pd.read_csv("Titanic-Dataset.csv")
df


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


num_col='Fare'


# In[8]:


mean_val=df[num_col].mean()
print('mean:',mean_val)


# In[9]:


mode_val=df[num_col].mode()
print('mode:',mode_val)


# In[10]:


stnad=df[num_col].std()
print('stnad:',stnad)


# In[11]:


variance=df[num_col].var()
print('var:',variance)


# In[12]:


range=df[num_col].max()-df[num_col].min()
print('range:',range)


# In[13]:


med_val=df[num_col].median()
print('median',med_val)


# In[21]:


pip install seaborn


# In[25]:


plt.figure(figsize=(10,5))
ns.distplot(df[num_col],bins=20,kde=True,color='pink')
plt.title(f"Histogram of{num_col}")
plt.xlabel("num_col")
plt.ylabel("frequency")
plt.show


# In[26]:


Q1=df[num_col].quantile(0.25)
Q3=df[num_col].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outlires=df[(df[num_col]<lower_bound)|(df[num_col]>upper_bound)]
print("\n outlires in the dataset")
print(outlires[num_col])


# In[27]:


category='Survived'


# In[28]:


cat_count=df[category].value_counts()
cat_count


# In[ ]:





# In[29]:


plt.figure(figsize=(6,6))
plt.pie(cat_count, labels=cat_count.index, colors=["pink", "red"])
plt.title(f"pie chart of {category}")
plt.show()


# In[30]:


plt.figure(figsize=(6,6))
plt.pie(cat_count, labels=cat_count.index,autopct="%1.1f%%" ,colors=["pink", "red"])
plt.title(f"pie chart of {category}")
plt.show()


# In[94]:


plt.figure(figsize=(10,5))
ns.boxplot(x=df[num_col],color='yellow')
plt.title(f"boxplot {num_col}")
plt.show()


# # program 2

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns


# In[66]:


d=pd.read_csv("iris.data.csv")
d


# In[67]:


d=pd.read_csv("iris.data.csv",header=None)
d


# In[68]:


d.isnull().sum()


# In[69]:


d.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
print( d)


# In[77]:


d1='sepal_length'
d2='sepal_width'


# In[79]:


plt.figure(figsize=(8,6))
ns.scatterplot(x=d[d1],y=d[d2],hue=d['speciecs'],palette='dark')
plt.title(f"Scatterplot of {d1} vs {d2}")
plt.xlabel(d1)
plt.ylabel(d2)
plt.show()


# In[83]:


pear_cor=d[d1].corr(d[d2])
print(f"Pearson Corelation coeffienct between {d1} and {d2}:",pear_cor)


# In[85]:


cov_matrix=d[[d1,d2]].cov()
print("\n Covarince Matrix:")
print(cov_matrix)


# In[86]:


corr_matrix=d.drop(columns=['speciecs']).corr()
print("\n Correlation Matrix:")
print(corr_matrix)


# In[92]:


plt.figure(figsize=(8,6))
ns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title(' Corelatin Matrix Heatmap')
plt.show()


# In[120]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns


# In[121]:


dn=pd.read_csv("Titanic-Dataset.csv")
dn


# In[122]:


dn1='PassengerId'
dn2='Fare'



# In[123]:


plt.figure(figsize=(8,6))
ns.scatterplot(x=dn[dn1],y=dn[dn2],hue=dn['Embarked'],palette='dark')
plt.title(f"Scatterplot of {d1} vs {d2}")
plt.xlabel(dn1)
plt.ylabel(dn2)
plt.show()


# In[125]:


pear_cor=dn[dn1].corr(dn[dn2])
print(f"Pearson Corelation coeffienct between {dn1} and {dn2}:",pear_cor)


# In[126]:


cov_matrix=dn[[dn1,dn2]].cov()
print("\n Covarince Matrix:")
print(cov_matrix)


# In[129]:


corr_matrix=dn.corr()
print("\n Correlation Matrix:")
print(corr_matrix)


# In[130]:


plt.figure(figsize=(8,6))
ns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title(' Corelatin Matrix Heatmap')
plt.show()


# # program 3

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[13]:


dn=pd.read_csv("iris.data.csv")
dn


# In[14]:


dn=pd.read_csv("iris.data.csv",header=None)
dn


# In[15]:


dn.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
print( dn)


# In[16]:


dn.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
dn.head(3)


# In[22]:


dn1=dn.drop(columns='speciecs')


# In[24]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(dn1)
x_scaled


# In[25]:


pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
x_pca


# In[31]:


dn2=pd.DataFrame(x_pca,columns=['PCA1','PCA2'])
dn2


# In[32]:


dn2['speciecs']=dn['speciecs']
dn2['speciecs']


# In[44]:


plt.figure(figsize=(8,6))
ns.scatterplot(x=dn2['PCA1'],y=dn2['PCA2'],hue=dn2['speciecs'],palette='dark')
plt.title(f" PCA from 4 to 2 features")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


# In[41]:


print("Variance ratio",pca.explained_variance_ratio_)


#  # program 4

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score


# In[2]:


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# In[3]:


data=load_iris()
data


# In[4]:


x=data.data
y=data.target


# In[5]:


print(x)


# In[6]:


print(y)


# In[7]:


std_scalar=StandardScaler()
x_scaled=std_scalar.fit_transform(x)
x_scaled


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[10]:


def evaluate_knn(k_values,weighted='False'):
    results=[]
    for k in k_values:
        if weighted:
            model=KNeighborsClassifier(n_neighbors=k,weights='distance')
        else:
            model=KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        acc=accuracy_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred,average='weighted')
        results.append((k,acc,f1))
        print(f"k=(k)|Accuracy:{acc:4f}|F1-score{f1:.4f}")
    return results 


# In[11]:


k_values=[1,3,5]
print("\n Regular KNN Results")
regular_knn_result=evaluate_knn(k_values,weighted='uniform')
print("\n weighted knn results")
weighted_knn_result=evaluate_knn(k_values,weighted=True)


# # pro 6

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[23]:


df = pd.read_csv("HousingData.csv")


# In[24]:


df


# In[25]:


x=df[['RM']].values
y=df['MEDV'].values


# In[26]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[27]:


def get_weights(X_train,query_point,tau):
    return np.exp(-np.sum((X_train - query_point) ** 2,axis=1)/(2*tau**2))


# In[35]:


def locally_weighted_regression(X_train,y_train,tau,X_test):
    y_pred=[]
    
    for x in X_test:
        weights=get_weights(X_train,x,tau)
        W = np.diag(weights)
        
        X_bias =np.c_[np.ones(X_train.shape[0]),X_train]
        theta=np.linalg.pinv(X_bias.T @ W @ X_bias)@ (X_bias.T @ W @ y_train)
        x_bias=np.array([1,x[0]])
        y_pred.append(x_bias @ theta)
    return np.array(y_pred)    


# In[36]:


X_test=np.linspace(min(x_scaled),max(x_scaled),100).reshape(-1,1)
tau_values=[0.1,0.5,1.0]


# In[37]:


plt.figure(figsize=(10,6))
plt.scatter(x_scaled,y,color='gray',label='Original Data')

for tau in tau_values:
    y_pred=locally_weighted_regression(x_scaled,y,tau,X_test)
    plt.plot(X_test,y_pred,label=f'LWR(t={tau})')
plt.xlabel('Scaled RM(Avg Rooms per Dweelling)')
plt.ylabel('House Price(MEDV)')
plt.title('Locally Weighted Regression on Boston Housing Dataset')
plt.legend()
plt.show()


# # program 7

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# In[7]:


df_boston=pd.read_csv("HousingData.csv")


# In[12]:


x_boston=df_boston[['RM']].values
y_boston=df_boston['MEDV'].values
model_linear=LinearRegression()
model_linear.fit(x_boston,y_boston)
x_test=np.linspace(min(x_boston),max(x_boston),100).reshape(-1,1)
y_pred=model_linear.predict(x_test)
plt.figure(figsize=(10,5))
plt.scatter(x_boston,y_boston,color='gray',label='Original Data')
plt.plot(x_test,y_pred,color='red',label='Linear Regression')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('House Price(MEDV)')
plt.title('Linear Regression - Boston Housing ')
plt.legend()
plt.show()


# In[13]:


url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df_mpg=pd.read_csv(url)
df_mpg=df_mpg.dropna()
x_mpg=df_mpg[['horsepower']].values
y_mpg=df_mpg['mpg'].values

scaler=StandardScaler()
x_mpg_scaled=scaler.fit_transform(x_mpg)

degrees=[2,3]
plt.figure(figsize=(10,5))
plt.scatter(x_mpg,y_mpg,color='gray',label='Original Data')

for d in degrees:
    model_poly=make_pipeline(PolynomialFeatures(d),LinearRegression())
    model_poly.fit(x_mpg_scaled,y_mpg)
    
    x_test_scaled=scaler.transform(np.linspace(min(x_mpg),max(x_mpg),100).reshape(-1,1))
    y_pred_poly=model_poly.predict(x_test_scaled)    
    plt.plot(np.linspace(min(x_mpg),max(x_mpg),100),y_pred_poly,label=f'Polynomial Regression(Degree{d})')
    
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomail Regression -Auto MPG')
plt.legend()
plt.show()   


# # program 8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_csv("Titanic-Dataset.csv")
df


# In[4]:


df=df[['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked']]
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
label_encoder=LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])
x=df.drop(columns=['Survived'])
y=df['Survived']


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[7]:


model=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=42)
model.fit(x_train,y_train)
plt.figure(figsize=(15,8))
plot_tree(model, feature_names=x.columns,class_names=['Not Survived','Survived'],filled=True)
plt.show()


# In[9]:


y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print(f"Accuracy:{accuracy:.2f}")
print(f"Precision:{precision:.2f}")
print(f"Recall:{recall:.2f}")
print(f"F1-Score:{f1:.2f}")

print("\nClassification Report:\n",classification_report(y_test,y_pred))


# In[ ]:




