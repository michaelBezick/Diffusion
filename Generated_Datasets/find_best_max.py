import os
import pdb

dir_list = os.listdir("./")
print(dir_list)

greatest_max = 0
greatest_folder = ""

for folder in dir_list:
    print(folder)
    if folder == "Experiment_-1" or folder =="find_best_max.py" or "." in folder:
        continue

    with open(folder + "/Experiment_Summary.txt", "r") as file:
       line = file.readline()
       print(line)
       max_gen = float(line.split(" ")[2])
       if max_gen > greatest_max:
           greatest_max = max_gen
           greatest_folder = folder

print(f"Greatest Folder: {greatest_folder}, max: {greatest_max}")
