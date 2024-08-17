from src.dataset.How2signDataset import How2signDataset
from src.utils.files import find_files

json_files = ["CO6qyvvglAE_18-5-rgb_front.json"]
csv_file = "how2sign_realigned_val.csv"
# json_files = find_files("data/how2sign/realigned_val", pattern='**/*.json', interval=1)
dataset = How2signDataset(json_files=json_files, csv_file=csv_file)
x, y = dataset.how2sign_keypoints_sentence()

print(x.shape)
print(y.shape)