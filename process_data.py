
import csv
import os
import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("log", help = "log file directory")
args = parser.parse_args()
cols = ['factor', 'answer_correct', 'episode_length', 'invalid_action_percent']
dfs = pd.DataFrame(columns=cols)

for file_name in glob.glob(os.path.join(args.log, '*.csv')):
	df = pd.read_csv(file_name, skiprows = 1, skipinitialspace=True)
	#df.rename(columns=lambda x: x.strip())
	file_name = os.path.splitext(file_name)[0]
	factor = file_name.split('_')[-1]
	
	data = {cols[i]:df[cols[i]].mean() for i in range(1, 4)}
	data[cols[0]] = factor
	dfs = dfs.append(data, ignore_index=True)
dfs = dfs.sort_values(by=cols[0])
print(dfs)

#question_type answer_correct answer gt_answer episode_length
#invalid_action_percent scene number seed required_interaction
#df = pd.read_csv(os.path.join(log_path, 't1s12345_0.2.csv'), skiprows = 1)
#print(df['question_type'])
"""
with open(os.path.join(log_path, 't1s12345_0.2.csv'), newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:
    print(row)
"""