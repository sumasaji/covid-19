{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2d6473bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0            6      148             72             35        0  33.6   \n",
       "1            1       85             66             29        0  26.6   \n",
       "2            8      183             64              0        0  23.3   \n",
       "3            1       89             66             23       94  28.1   \n",
       "4            0      137             40             35      168  43.1   \n",
       "\n",
       "   DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                     0.627   50        1  \n",
       "1                     0.351   31        0  \n",
       "2                     0.672   32        1  \n",
       "3                     0.167   21        0  \n",
       "4                     2.288   33        1  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "df=pd.read_csv('diabetes.csv')\n",
    "df.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "795a692f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(768, 9)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "22710d27",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 768 entries, 0 to 767\n",
      "Data columns (total 9 columns):\n",
      " #   Column                    Non-Null Count  Dtype  \n",
      "---  ------                    --------------  -----  \n",
      " 0   Pregnancies               768 non-null    int64  \n",
      " 1   Glucose                   768 non-null    int64  \n",
      " 2   BloodPressure             768 non-null    int64  \n",
      " 3   SkinThickness             768 non-null    int64  \n",
      " 4   Insulin                   768 non-null    int64  \n",
      " 5   BMI                       768 non-null    float64\n",
      " 6   DiabetesPedigreeFunction  768 non-null    float64\n",
      " 7   Age                       768 non-null    int64  \n",
      " 8   Outcome                   768 non-null    int64  \n",
      "dtypes: float64(2), int64(7)\n",
      "memory usage: 54.1 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b8ea6e62",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pregnancies                 0\n",
       "Glucose                     0\n",
       "BloodPressure               0\n",
       "SkinThickness               0\n",
       "Insulin                     0\n",
       "BMI                         0\n",
       "DiabetesPedigreeFunction    0\n",
       "Age                         0\n",
       "Outcome                     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1c13c7ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "41646854",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.iloc[:,df.columns!='Outcome ']\n",
    "y=df.iloc[:,df.columns=='Outcome '] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "ca2d49ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
      "0              6      148             72             35        0  33.6   \n",
      "1              1       85             66             29        0  26.6   \n",
      "2              8      183             64              0        0  23.3   \n",
      "3              1       89             66             23       94  28.1   \n",
      "4              0      137             40             35      168  43.1   \n",
      "..           ...      ...            ...            ...      ...   ...   \n",
      "763           10      101             76             48      180  32.9   \n",
      "764            2      122             70             27        0  36.8   \n",
      "765            5      121             72             23      112  26.2   \n",
      "766            1      126             60              0        0  30.1   \n",
      "767            1       93             70             31        0  30.4   \n",
      "\n",
      "     DiabetesPedigreeFunction  Age  Outcome  \n",
      "0                       0.627   50        1  \n",
      "1                       0.351   31        0  \n",
      "2                       0.672   32        1  \n",
      "3                       0.167   21        0  \n",
      "4                       2.288   33        1  \n",
      "..                        ...  ...      ...  \n",
      "763                     0.171   63        0  \n",
      "764                     0.340   27        0  \n",
      "765                     0.245   30        0  \n",
      "766                     0.349   47        1  \n",
      "767                     0.315   23        0  \n",
      "\n",
      "[768 rows x 9 columns]\n"
     ]
    }
   ],
   "source": [
    "print(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "49b33086",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: []\n",
      "Index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, ...]\n",
      "\n",
      "[768 rows x 0 columns]\n"
     ]
    }
   ],
   "source": [
    "print(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "f8a69338",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "04d4ad94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>653</th>\n",
       "      <td>2</td>\n",
       "      <td>120</td>\n",
       "      <td>54</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>26.8</td>\n",
       "      <td>0.455</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>474</th>\n",
       "      <td>4</td>\n",
       "      <td>114</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28.9</td>\n",
       "      <td>0.126</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>320</th>\n",
       "      <td>4</td>\n",
       "      <td>129</td>\n",
       "      <td>60</td>\n",
       "      <td>12</td>\n",
       "      <td>231</td>\n",
       "      <td>27.5</td>\n",
       "      <td>0.527</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>480</th>\n",
       "      <td>3</td>\n",
       "      <td>158</td>\n",
       "      <td>70</td>\n",
       "      <td>30</td>\n",
       "      <td>328</td>\n",
       "      <td>35.5</td>\n",
       "      <td>0.344</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>218</th>\n",
       "      <td>5</td>\n",
       "      <td>85</td>\n",
       "      <td>74</td>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>1.224</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "653            2      120             54              0        0  26.8   \n",
       "474            4      114             64              0        0  28.9   \n",
       "320            4      129             60             12      231  27.5   \n",
       "480            3      158             70             30      328  35.5   \n",
       "218            5       85             74             22        0  29.0   \n",
       "\n",
       "     DiabetesPedigreeFunction  Age  Outcome  \n",
       "653                     0.455   27        0  \n",
       "474                     0.126   24        0  \n",
       "320                     0.527   31        0  \n",
       "480                     0.344   35        1  \n",
       "218                     1.224   32        1  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xtrain.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "1a2105a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>653</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>474</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>320</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>480</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>218</th>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: []\n",
       "Index: [653, 474, 320, 480, 218]"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ytrain.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "eeeae2c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d7e000f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "db5da45f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "787fce2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(614, 0)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ytrain.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "d7a51059",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('Outcome', axis=1)\n",
    "y = df['Outcome']\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Create a Random Forest Classifier\n",
    "model = RandomForestClassifier()\n",
    "\n",
    "# Train the model\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test set\n",
    "xtrain = X \n",
    "ytest = y  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d712c900",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 1 0 0 0 1 1 0 0 0 0 0\n",
      " 1 1 1 0 0 0 1 0 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0\n",
      " 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1\n",
      " 1 0 0 1 1 1 0 0 0 1 0 0 0 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0\n",
      " 1 0 0 0 1 0 1 1 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0\n",
      " 1 1 1 1 1 0 0 1 1 0 1 0 1 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0 1 1 1 0 0 1 1 1\n",
      " 0 0 0 0 0 1 0 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 1 0 0 0\n",
      " 1 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 0 0 0 1 1 1 0 0\n",
      " 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 0 0 1 0 0 0 0 0 1\n",
      " 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 1 1 1 0 1 0 0 1 0 0 1\n",
      " 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 1 1 0 1 0 1 0 1\n",
      " 0 1 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1\n",
      " 1 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1\n",
      " 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 1 0\n",
      " 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 1 1 0 0 0 0 0\n",
      " 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 0 1 0 1 0 1 0\n",
      " 1 0 0 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0\n",
      " 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 0\n",
      " 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 0 1 1\n",
      " 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 1\n",
      " 1 0 0 1 0 0 1 0 1 1 1 0 0 1 1 1 0 1 0 1 0 1 0 0 0 0 1 0]\n"
     ]
    }
   ],
   "source": [
    "predict_output=model.predict(xtrain)\n",
    "print(predict_output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "593ad6e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "380422c0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The accuracy_score for Random Forest: 0.9505208333333334\n"
     ]
    }
   ],
   "source": [
    "ac = accuracy_score(predict_output, ytest)\n",
    "print(\"The accuracy_score for Random Forest:\", ac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "252c407f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
