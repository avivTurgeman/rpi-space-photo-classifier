# %%
import os
import data_utils

cat_ = "stars"
# root_path = Path(__file__).resolve().parents[1]
dir_ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
# print(dir_)
train_, val_, test_ = data_utils.get_datasets_for_category(dir_, cat_)
print(train_)
# %%
