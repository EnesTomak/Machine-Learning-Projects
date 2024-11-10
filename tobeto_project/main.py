import pandas as pd
import numpy as np
import warnings

import pydotplus as pydotplus
# görselleştirme kütüphaneleri
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image

# makine öğrenimi kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Ayarların yapılandırılması
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action="ignore", category=Warning)


df_application_record = pd.read_csv("datasets/application_record.csv")

df_credit_record = pd.read_csv("datasets/application_record.csv")


df = pd.merge(df_application_record, df_credit_record, on='ID', how='inner')


def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    Veri setindeki kategorik, sayısal ve kategorik ama kardinal değişkenleri ayırır.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# Eksik değer
def find_and_drop_missing_values(df):
    missing_valuable_columns = df.columns[df.isna().sum() > (df.shape[0] * 0.10)]
    for col in missing_valuable_columns:
        missing_value_ratio = (df[col].isna().sum() / df.shape[0]) * 100
        print(f"{col} sütununda eksik değer oranı: %{missing_value_ratio:.2f}")
    df.drop(missing_valuable_columns, axis=1, inplace=True)
    return df


# Sütunları yeniden adlandırma
def rename_columns(df):
    columns_mapping = {
        'CODE_GENDER': 'gender',
        'FLAG_OWN_CAR': 'own_car',
        'FLAG_OWN_REALTY': 'own_property',
        'CNT_CHILDREN': 'children',
        'AMT_INCOME_TOTAL': 'income',
        'NAME_INCOME_TYPE': 'income_type',
        'NAME_EDUCATION_TYPE': 'education',
        'NAME_FAMILY_STATUS': 'family_status',
        'NAME_HOUSING_TYPE': 'housing_type',
        'FLAG_MOBIL': 'mobile',
        'FLAG_WORK_PHONE': 'work_phone',
        'FLAG_PHONE': 'phone',
        'FLAG_EMAIL': 'email',
        'CNT_FAM_MEMBERS': 'family_members',
        'MONTHS_BALANCE': 'months_balance',
        'STATUS': 'status',
        'DAYS_BIRTH': 'age_in_days',
        'DAYS_EMPLOYED': 'employment_in_days'
    }
    df.rename(columns=columns_mapping, inplace=True)
    return df
rename_columns(df)



# aykırı değerler değiştirildi
def replace_outliers(df):
    col = ['children', 'income', 'family_members']
    for i in range(len(col)):
        q1 = df[col[i]].quantile(0.15)
        q3 = df[col[i]].quantile(0.85)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col[i]] = np.where((df[col[i]] >= upper_bound) | (df[col[i]] <= lower_bound), df[col[i]].median(), df[col[i]])
    return df

# Kodlama ve ölçeklendirme
def encode_and_scale(df, cat_cols, num_cols):
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    # Encoding the 'INCOME_CATEGORY' column if it exists
    if 'INCOME_CATEGORY' in df.columns:
        df['INCOME_CATEGORY'] = le.fit_transform(df['INCOME_CATEGORY'])
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

df = find_and_drop_missing_values(df)
df = rename_columns(df)
df = replace_outliers(df)

# gender benzersiz girişlerinin haritalanması
df['gender'] = df['gender'].map({'F': 'female', 'M': 'male'})

# own_car'ın benzersiz girişlerini eşleme
df['own_car'] = df['own_car'].map({'N': 'no', 'Y': 'yes'})

# own_property'nin benzersiz girişlerini eşleme
df['own_property'] = df['own_property'].map({'N': 'no', 'Y': 'yes'})

# Özellik Mühendisliği

# hedef değişkende ki değerleri  anlamlı şekilde kategorize etme
df['loan_status'] = df['status'].map({'0': 'first_month_due', '1': '2nd_month_due', '2': '3rd_month_overdue',
                                      '3': '4th_month_overdue', '4': '5th_month_overdue', '5': 'bad_debt',
                                      'C': 'good', 'X': 'no_loan'})

# Gelir aralıklarını ve etiketleri belirleme
bins = [0, 75000, 150000, 250000, 400000, 600000]
labels = ['Low', 'Lower Middle', 'Middle', 'Upper Middle', 'High']

df['INCOME_CATEGORY'] = pd.cut(df['income'], bins=bins, labels=labels)

# Çocuk başına düşen gelir
df['income_per_child'] = df['income'] / (df['children'] + 1)

# Aile başına düşen gelir
df['income_per_family_member'] = df['income'] / df['family_members']

# Mevcut işverende çalışılan yıl sayısı
df['years_employed'] = df['employment_in_days']//-365

# Çoklu doğrusallığı azaltmak için etkileyen değişkenleri silindi
df.drop(['employment_in_days'], axis=1, inplace=True)
df.drop(['children'], axis=1, inplace=True)
df.drop(['income'], axis=1, inplace=True)
df.drop(['mobile'], axis=1, inplace=True)
df.drop(['family_members'], axis=1, inplace=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = encode_and_scale(df, cat_cols, num_cols)



# Varyans Enflasyon Faktörü (VIF)
def calculate_vif(df):
    # Select only the numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Drop 'loan_status' and 'status' columns if they are in the DataFrame
    if 'loan_status' in numeric_df.columns:
        numeric_df = numeric_df.drop(['loan_status'], axis=1)
    if 'status' in numeric_df.columns:
        numeric_df = numeric_df.drop(['status'], axis=1)

    # Create a DataFrame to store the VIF values
    factor = pd.DataFrame()
    factor["Features"] = numeric_df.columns
    factor["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]

    return factor

vif_factors = calculate_vif(df)
print(vif_factors)

# Modeli kurma

X = df.drop(['loan_status', 'status'], axis=1)
y = df['loan_status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Modelin tahmin etmesi
y_pred = model.predict(X_test)

# Gerçekleşen ve tahmin edilen değerleri karşılaştırmak
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head(10))
print(results_df.tail(10))

# Model Değerlendirmesi

print(f'Accuracy Score : {accuracy_score(y_test, y_pred)*100:.2f} %')

print(f"Precision Score : {precision_score(y_test, y_pred , average = 'micro'):.2f}")

print(f"F1-Score : {f1_score(y_test, y_pred , average = 'micro'):.2f}")

# sınıflandırma raporu
print(classification_report(y_test, y_pred))

