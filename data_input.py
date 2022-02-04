from pandas import to_datetime
from pandas import read_csv
from config import date_train_init, date_train_end

# data input {
# =============================================================================

# data {
# -----------------------------------------------------------------------------
data_entry = read_csv("1_data/0_data_base.csv", sep=",", decimal=".")
data_entry["index_date"] = to_datetime(data_entry["index_date"])
data_entry = data_entry.sort_values("index_date")

# date original
data_original = data_entry[ (data_entry.iloc[:,0] >= date_train_init) ]
data_original = data_original.set_index("index_date")
data_original = data_original.iloc[:,0]

# date train
data_train = data_entry[ (data_entry.iloc[:,0] >= date_train_init) ]
data_train = data_train.set_index("index_date")

# data variables
data_endog = data_train.iloc[ : , 0 : 1 ]
data_exogs = data_train.iloc[ : , 1 :   ]

# -----------------------------------------------------------------------------
#   }

# variables {
# -----------------------------------------------------------------------------
variable_ = list(data_endog.columns.values.tolist())[0]
variable = variable_.replace("_", ' ').upper()
# -----------------------------------------------------------------------------
#   }

# =============================================================================
# }
