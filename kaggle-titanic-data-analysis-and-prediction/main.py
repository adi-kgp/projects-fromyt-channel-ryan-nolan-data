# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# importing data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


"""--------------- Exploratory Data Analysis------------------"""

print(train_df.info())

print(train_df.describe())
# categorical data description
print(train_df.describe(include=['O']))

print(train_df.groupby(['Pclass'], as_index=False)['Survived'].mean())
print(train_df.groupby(['Sex'], as_index=False)['Survived'].mean())
print(train_df.groupby(['SibSp'], as_index=False)['Survived'].mean())
print(train_df.groupby(['Parch'], as_index=False)['Survived'].mean())

train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch'] + 1

print(train_df.groupby(['Family_Size'], as_index=False)['Survived'].mean())

# changing numerical variable to categorical
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium',
              6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

train_df['Family_Size_Grouped'] = train_df['Family_Size'].map(family_map)
test_df['Family_Size_Grouped'] = test_df['Family_Size'].map(family_map)

print(train_df.groupby(['Family_Size_Grouped'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Embarked'], as_index=False)['Survived'].mean())


# Visualization

sns.displot(train_df, x='Age', col='Survived', binwidth=10, height=5)

train_df['Age_Cut'] = pd.qcut(train_df['Age'], 8)
test_df['Age_Cut'] = pd.qcut(test_df['Age'], 8)

print(train_df.groupby(['Age_Cut'], as_index=False)['Survived'].mean())


# changing the age column from numerical to categorical
train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 20.125), 'Age'] = 1
train_df.loc[(train_df['Age'] > 20.125) & (train_df['Age'] <= 24.0), 'Age'] = 2
train_df.loc[(train_df['Age'] > 24.0) & (train_df['Age'] <= 28.0), 'Age'] = 3
train_df.loc[(train_df['Age'] > 28.0) & (train_df['Age'] <= 32.312), 'Age'] = 4
train_df.loc[(train_df['Age'] > 32.312) & (train_df['Age'] <= 38.0), 'Age'] = 5
train_df.loc[(train_df['Age'] > 38.0) & (train_df['Age'] <= 47.0), 'Age'] = 6
train_df.loc[(train_df['Age'] > 47.0) & (train_df['Age'] <= 80.0), 'Age'] = 7
train_df.loc[train_df['Age'] > 80, 'Age'] 


test_df.loc[test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 20.125), 'Age'] = 1
test_df.loc[(test_df['Age'] > 20.125) & (test_df['Age'] <= 24.0), 'Age'] = 2
test_df.loc[(test_df['Age'] > 24.0) & (test_df['Age'] <= 28.0), 'Age'] = 3
test_df.loc[(test_df['Age'] > 28.0) & (test_df['Age'] <= 32.312), 'Age'] = 4
test_df.loc[(test_df['Age'] > 32.312) & (test_df['Age'] <= 38.0), 'Age'] = 5
test_df.loc[(test_df['Age'] > 38.0) & (test_df['Age'] <= 47.0), 'Age'] = 6
test_df.loc[(test_df['Age'] > 47.0) & (test_df['Age'] <= 80.0), 'Age'] = 7
test_df.loc[test_df['Age'] > 80, 'Age'] 


sns.displot(train_df, x='Fare', col='Survived', binwidth=80, height=5)

train_df['Fare_Cut'] = pd.qcut(train_df['Fare'], 6)
test_df['Fare_Cut'] = pd.qcut(test_df['Fare'], 6)

print(train_df.groupby(['Fare_Cut'], as_index=False)['Survived'].mean())


# changing the Fare column from numerical to categorical
train_df.loc[train_df['Fare'] <= 7.775, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.775) & (train_df['Fare'] <= 8.662), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 8.662) & (train_df['Fare'] <= 14.454), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 26.0), 'Fare'] = 3
train_df.loc[(train_df['Fare'] > 26.0) & (train_df['Fare'] <= 52.369), 'Fare'] = 4
train_df.loc[(train_df['Fare'] > 52.369) & (train_df['Fare'] <= 512.329), 'Fare'] = 5
train_df.loc[train_df['Fare'] > 512.329, 'Fare'] 


test_df.loc[test_df['Fare'] <= 7.775, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.775) & (test_df['Fare'] <= 8.662), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 8.662) & (test_df['Fare'] <= 14.454), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 26.0), 'Fare'] = 3
test_df.loc[(test_df['Fare'] > 26.0) & (test_df['Fare'] <= 52.369), 'Fare'] = 4
test_df.loc[(test_df['Fare'] > 52.369) & (test_df['Fare'] <= 512.329), 'Fare'] = 5
test_df.loc[test_df['Fare'] > 512.329, 'Fare'] 


train_df['Title'] = train_df['Name'].str.split(pat=",", expand=True)[1].str.split(pat=".", expand=True)[0].apply(lambda x: x.strip())
test_df['Title'] = test_df['Name'].str.split(pat=",", expand=True)[1].str.split(pat=".", expand=True)[0].apply(lambda x: x.strip())

print(train_df.groupby(['Title'], as_index=False)['Survived'].mean())

"""
military: Capt, Col, Major
noble: Jonkheer, the Countess, Don , Lady, Sir
unmarried Female: Mlle, Ms, Mme
"""

train_df['Title'] = train_df['Title'].replace({
        'Capt': 'Military',
        'Col': 'Military',
        'Major': 'Military',
        'Jonkheer': 'Noble',
        'the Countess': 'Noble',
        'Don': 'Noble',
        'Lady': 'Noble',
        'Sir': 'Noble',
        'Mlle': 'Noble',
        'Ms': 'Noble',
        'Mme': 'Noble'
    })

test_df['Title'] = test_df['Title'].replace({
        'Capt': 'Military',
        'Col': 'Military',
        'Major': 'Military',
        'Jonkheer': 'Noble',
        'the Countess': 'Noble',
        'Don': 'Noble',
        'Lady': 'Noble',
        'Sir': 'Noble',
        'Mlle': 'Noble',
        'Ms': 'Noble',
        'Mme': 'Noble'
    })


print(train_df.groupby(['Title'], as_index=False)['Survived'].agg(['count', 'mean']))


train_df['Name_Length'] = train_df['Name'].apply(lambda x: len(x))
test_df['Name_Length'] = test_df['Name'].apply(lambda x: len(x))


g = sns.kdeplot(train_df['Name_Length'][(train_df['Survived']==0) & (train_df['Name_Length'].notnull())], color='Red', fill=True)
g = sns.kdeplot(train_df['Name_Length'][(train_df['Survived']==1) & (train_df['Name_Length'].notnull())], ax=g, color='Blue', fill=True)

g.set_xlabel('Name_Length')
g.set_ylabel('Frequency')
g = g.legend(['Not Survived', 'Survived'])


train_df['Name_LengthGB'] = pd.qcut(train_df['Name_Length'], 8)
test_df['Name_LengthGB'] = pd.qcut(test_df['Name_Length'], 8)


print(train_df.groupby(['Name_LengthGB'], as_index=False)['Survived'].mean())

# Changing the Name_Length column from numerical to categorical
train_df.loc[train_df['Name_Length'] <= 18.0, 'Name_Size'] = 0
train_df.loc[(train_df['Name_Length'] > 18) & (train_df['Name_Length'] <= 20.0), 'Name_Size'] = 1
train_df.loc[(train_df['Name_Length'] > 20.0) & (train_df['Name_Length'] <= 23.0), 'Name_Size'] = 2
train_df.loc[(train_df['Name_Length'] > 23.0) & (train_df['Name_Length'] <= 25.0), 'Name_Size'] = 3
train_df.loc[(train_df['Name_Length'] > 25.0) & (train_df['Name_Length'] <= 27.25), 'Name_Size'] = 4
train_df.loc[(train_df['Name_Length'] > 27.25) & (train_df['Name_Length'] <= 30.0), 'Name_Size'] = 5
train_df.loc[(train_df['Name_Length'] > 30.0) & (train_df['Name_Length'] <= 38.0), 'Name_Size'] = 6
train_df.loc[(train_df['Name_Length'] > 38.0) & (train_df['Name_Length'] <= 82.0), 'Name_Size'] = 7
train_df.loc[train_df['Name_Length'] > 82.0, 'Name_Length'] 


test_df.loc[test_df['Name_Length'] <= 18, 'Name_Size'] = 0
test_df.loc[(test_df['Name_Length'] > 18.0) & (test_df['Name_Length'] <= 20.0), 'Name_Size'] = 1
test_df.loc[(test_df['Name_Length'] > 20.0) & (test_df['Name_Length'] <= 23.0), 'Name_Size'] = 2
test_df.loc[(test_df['Name_Length'] > 23.0) & (test_df['Name_Length'] <= 25.0), 'Name_Size'] = 3
test_df.loc[(test_df['Name_Length'] > 25.0) & (test_df['Name_Length'] <= 27.25), 'Name_Size'] = 4
test_df.loc[(test_df['Name_Length'] > 27.25) & (test_df['Name_Length'] <= 30.0), 'Name_Size'] = 5
test_df.loc[(test_df['Name_Length'] > 30.0) & (test_df['Name_Length'] <= 38.0), 'Name_Size'] = 6
test_df.loc[(test_df['Name_Length'] > 38.0) & (test_df['Name_Length'] <= 82.0), 'Name_Size'] = 7
test_df.loc[test_df['Name_Length'] > 82.0, 'Name_Length'] 


# ticket column
train_df['TicketNumber'] = train_df['Ticket'].apply(lambda x: pd.Series({'Ticket': x.split()[-1]}))
test_df['TicketNumber'] = test_df['Ticket'].apply(lambda x: pd.Series({'Ticket': x.split()[-1]}))

print(train_df.groupby(['TicketNumber'], as_index=False)['Survived'].agg(['count', 'mean']).sort_values('count', ascending=False))

train_df.groupby('TicketNumber')['TicketNumber'].transform('count')

train_df['TicketNumberCounts'] = train_df.groupby('TicketNumber')['TicketNumber'].transform('count')
test_df['TicketNumberCounts'] = test_df.groupby('TicketNumber')['TicketNumber'].transform('count')

print(train_df.groupby(['TicketNumberCounts'], as_index=False)['Survived'].agg(['count', 'mean']).sort_values('count', ascending=False))


train_df['TicketLocation'] = np.where(train_df['Ticket'].str.split(pat=" ", expand=True)[1].notna(),\
         train_df['Ticket'].str.split(pat=" ", \
                                      expand=True)[0].apply(lambda x: x.strip()), 'Blank')
    
test_df['TicketLocation'] = np.where(test_df['Ticket'].str.split(pat=" ", expand=True)[1].notna(),\
         test_df['Ticket'].str.split(pat=" ", \
                                      expand=True)[0].apply(lambda x: x.strip()), 'Blank')

print(train_df['TicketLocation'].value_counts())


train_df['TicketLocation'] = train_df['TicketLocation'].replace({
        'SOTON/O.Q.':'SOTON/OQ',
        'C.A.':'CA',
        'CA.':'CA',
        'SC/PARIS':'SC/Paris',
        'S.C./PARIS': 'SC/Paris',
        'A/4.':'A/4',
        'A/5.':'A/5',
        'A.5.':'A/5',
        'A./5.':'A/5',
        'W./C.':'W/C'
    })

test_df['TicketLocation'] = test_df['TicketLocation'].replace({
        'SOTON/O.Q.':'SOTON/OQ',
        'C.A.':'CA',
        'CA.':'CA',
        'SC/PARIS':'SC/Paris',
        'S.C./PARIS': 'SC/Paris',
        'A/4.':'A/4',
        'A/5.':'A/5',
        'A.5.':'A/5',
        'A./5.':'A/5',
        'W./C.':'W/C'
    })

print(train_df.groupby(['TicketLocation'], as_index=False)['Survived'].agg(['count', 'mean']).sort_values('count', ascending=False))


# feature engineering Cabin column
train_df['Cabin'] = train_df['Cabin'].fillna('U')
train_df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'x' for i in train_df['Cabin']]) 

test_df['Cabin'] = test_df['Cabin'].fillna('U')
test_df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'x' for i in test_df['Cabin']]) 


print(train_df.groupby(['Cabin'], as_index=False)['Survived'].agg(['count', 'mean']))

train_df['Cabin_Assigned'] = train_df['Cabin'].apply(lambda x: 0 if x in ['U'] else 1)
test_df['Cabin_Assigned'] = test_df['Cabin'].apply(lambda x: 0 if x in ['U'] else 1)

print(train_df.groupby(['Cabin_Assigned'], as_index=False)['Survived'].agg(['count', 'mean']))


"""--------------Model Building--------------"""

print(train_df.info())
print(test_df.info())
print(train_df.columns)

train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)

# one hot encoding
ohe = OneHotEncoder(sparse_output=False)
ode = OrdinalEncoder
SI = SimpleImputer(strategy='most_frequent')

ode_cols = ['Family_Size_Grouped']
ohe_cols = ['Sex', 'Embarked']

X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

X_test = test_df.drop(['Age_Cut', 'Fare_Cut'], axis=1)

# splitting the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, 
                                                      stratify=y,
                                                      random_state=21)


# Pipelines 
ordinal_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

ohe_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown = 'ignore', sparse_output=False))
    ])

col_trans = ColumnTransformer(transformers=[
        ('impute', SI, ['Age']),
        ('ord_pipeline', ordinal_pipeline, ode_cols),
        ('ohe_pipeline', ohe_pipeline, ohe_cols),
        ('passthrough', 'passthrough', ['Pclass', 'TicketNumberCounts', 'Cabin_Assigned', 'Name_Size', 'Age', 'Fare'])
    ],
        remainder = 'drop',
        n_jobs = -1)


# model building

# random forest
rfc = RandomForestClassifier()

param_grid = {
        'n_estimators': [100, 150, 200],
        'min_samples_split': [5, 10, 15],
        'max_depth': [8, 9, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
    }

CV_rfc = GridSearchCV(rfc, param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalrfc = make_pipeline(col_trans, CV_rfc)
pipefinalrfc.fit(X_train, y_train)

print(CV_rfc.best_params_)
print(CV_rfc.best_score_)


# decision tree
dtc = DecisionTreeClassifier()

param_grid = {
        'min_samples_split': [5, 10, 15],
        'max_depth': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
    }

CV_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinaldtc = make_pipeline(col_trans, CV_dtc)
pipefinaldtc.fit(X_train, y_train)

print(CV_dtc.best_params_)
print(CV_dtc.best_score_)


# KNN
knn = KNeighborsClassifier()

param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1,2],
    }

CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalknn = make_pipeline(col_trans, CV_knn)
pipefinalknn.fit(X_train, y_train)

print(CV_knn.best_params_)
print(CV_knn.best_score_)


# Support Vector Machine
svc = SVC()

param_grid = {
        'C': [100,10, 1.0, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }

CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalsvc = make_pipeline(col_trans, CV_svc)
pipefinalsvc.fit(X_train, y_train)

print(CV_svc.best_params_)
print(CV_svc.best_score_)


# Logistic Regression
lr = LogisticRegression()

param_grid = {
        'C': [100,10, 1.0, 0.1, 0.01, 0.001],
    }

CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinallr = make_pipeline(col_trans, CV_lr)
pipefinallr.fit(X_train, y_train)

print(CV_lr.best_params_)
print(CV_lr.best_score_)


# Naive Bayes
gnb = GaussianNB()

param_grid = {
        'var_smoothing': [0.00000001, 0.000000001],
    }

CV_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalgnb = make_pipeline(col_trans, CV_gnb)
pipefinalgnb.fit(X_train, y_train)

print(CV_gnb.best_params_)
print(CV_gnb.best_score_)


# Predictions
Y_pred = pipefinalrfc.predict(X_test)
Y_pred2 = pipefinaldtc.predict(X_test)
Y_pred3 = pipefinalknn.predict(X_test)
Y_pred4 = pipefinalsvc.predict(X_test)
Y_pred5 = pipefinallr.predict(X_test)
Y_pred6 = pipefinalgnb.predict(X_test)
