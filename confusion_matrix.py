import random 

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
number_of_classes = len(classes)

row = []
row_str = ""

TP = "TP"
TN = "TN"
FN = "FN"
FP = "FP"
space = " "
padding = 1

OvR = classes[9]

threshold = 100

num_TP = 0
num_TN = 0
num_FP = 0
num_FN = 0

input = [[random.randint(0, threshold) for _ in range(number_of_classes)] for _ in range(number_of_classes)]

for i in classes:
    r = []
    for j in classes:
        if (i == OvR and j == OvR):
            row_str += TP + padding*space
            r.append(TP)
        elif (i != OvR and j != OvR):
            row_str += TN + padding*space
            r.append(TN)
        elif (i != OvR and j == OvR):
            row_str += FP + padding*space
            r.append(FP)
        elif (i == OvR and j != OvR):
            row_str += FN + padding*space
            r.append(FN)
    row.append(r)
    row_str += "\n"

for i in range(len(row)):
    for j in range(len(row[0])):
        if row[i][j] == TP:
            num_TP += input[i][j]
        elif row[i][j] == TN:
            num_TN += input[i][j]
        elif row[i][j] == FP:
            num_FP += input[i][j]
        elif row[i][j] == FN:
            num_FN += input[i][j]
        else:
            raise TypeError("Unknown Value, Supported Values are : TP, TN, FP, FN")

print("\nConfusion Matrix : \n")
print(row_str)

print("\n Error analysis : \n")
print(f"False Negatives : {num_FN}, False Positives : {num_FP}, True Negatives : {num_TN}, True Positives : {num_TP}")

accuracy = (num_TP + num_TN) / (num_TN + num_FN + num_FP + num_TP)
percision = num_TP / (num_FP + num_TP)
recall = num_TP / (num_TP + num_FN)
f1_score = 2 * (percision * recall) / (percision + recall)

print(f"Accuracy : {accuracy}")
print(f"Percision : {percision}")
print(f"Recall : {recall}")
print(f"F1 Score : {f1_score}")