
'''
Census Dataset

label:
income_bracket:  >50K, <=50K.

features:
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
'''

## feat config
CSV_HEADER = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket',
]

NUMERIC_FEATURE_NAMES = [
    'age',
    'education_num',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'fnlwgt',
]

CATEGORICAL_FEATURE_NAMES = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country'
]
TARGET_COL_NAME = 'income_bracket'
TARGET_LABELS = [' <=50K', ' >50K']

## model config
NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
EMBEDDING_DIMS = 16
# MLP_HIDDEN_UNITS_FACTORS = [1, 1,]
MLP_HIDDEN_UNITS_FACTORS = [2, 1,]
NUM_MLP_BLOCKS = 2

## training config
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.0001
DROPOUT_RATE  = 0.2
BATCH_SIZE    = 1024
NUM_EPOCHS    = 30
SEED          = 2021

