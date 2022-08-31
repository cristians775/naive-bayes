import pandas as pd
import matplotlib.pyplot as plt
from bayes import Bayes

df = pd.read_excel("./datos_de_entrenamiento.xlsx")
df2 = pd.read_excel("./datos_de_prueba.xlsx")

data_classification = df.to_numpy()
test_data_1 = df2.to_numpy()


attributes = [
    {1: {1: 0,
         2: 0,
         3: 0,
         4: 0
         },
     0: {
        1: 0,
         2: 0,
         3: 0,
        4: 0



    }},
    {1: {1: 0,
         2: 0,
         3: 0,
         4: 0
         },
     0: {
        1: 0,
        2: 0,
        3: 0,
        4: 0

    }},
    {1: {1: 0,
         2: 0,
         3: 0,
         4: 0
         },
     0: {
        1: 0,
         2: 0,
         3: 0,
         4: 0


    }},
    {1: {1: 0,
         2: 0,
         3: 0,
         4: 0
         },
     0: {
         1: 0,
         2: 0,
         3: 0,
         4: 0

    }},
    {1: {1: 0,
         2: 0,
         3: 0,
         4: 0
         },
     0: {
        1: 0,
         2: 0,
         3: 0,
         4: 0

    }}
]


def get_credit_ability_column(arr):
    return list(map(lambda element: element[0], arr))


def calculate_metrics(arr_1, arr_2):
    tp, tn, fp, fn = 0, 0, 0, 0
    result = {}
    for i in range(len(arr_1)):
        print(arr_1[i] == 0 and arr_2[i] == 0)
        if (arr_1[i] == 1 and arr_2[i] == 1):
            tp += 1
        elif (arr_1[i] == 0 and arr_2[i] == 0):
            tn += 1
        elif ((arr_1[i] == 0 and arr_2[i] == 1)):
            fp += 1
        elif ((arr_1[i] == 1 and arr_2[i] == 0)):
            fn += 1
    result['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
    result['precision'] = tp/(tp+fp)
    result['recall'] = tp/(tp+fn)
    result['f1_score'] = (2*result['precision']*result['recall']) / \
        (result['precision'] + result['recall'])
    result['tp_rate'] = tp/(tp+fn)
    result['fp_rate'] = fp/(tn+fp)
    result['tp'] = tp
    result['tn'] = tn
    result['fn'] = fn
    result['fp'] = fp
    return result


bayes = Bayes(attributes, data_classification)
bayes.train()


true_class_test_data_1 = get_credit_ability_column(test_data_1)
result_test_data = bayes.test(test_data_1, 0.05)
result_test_data_1 = bayes.test(test_data_1, 0.1)
result_test_data_2 = bayes.test(test_data_1, 0.2)
result_test_data_3 = bayes.test(test_data_1, 0.3)
result_test_data_4 = bayes.test(test_data_1, 0.4)
result_test_data_5 = bayes.test(test_data_1, 0.5)
result_test_data_6 = bayes.test(test_data_1, 0.6)
result_test_data_7 = bayes.test(test_data_1, 0.7)
result_test_data_8 = bayes.test(test_data_1, 0.8)
result_test_data_9 = bayes.test(test_data_1, 0.9)
p4 = bayes.test([[1, 3, 1, 3, 2]], 0.70)


result = calculate_metrics(true_class_test_data_1, result_test_data)
result_1 = calculate_metrics(true_class_test_data_1, result_test_data_1)
result_2 = calculate_metrics(true_class_test_data_1, result_test_data_2)
result_3 = calculate_metrics(true_class_test_data_1, result_test_data_3)
result_4 = calculate_metrics(true_class_test_data_1, result_test_data_4)
result_5 = calculate_metrics(true_class_test_data_1, result_test_data_5)
result_6 = calculate_metrics(true_class_test_data_1, result_test_data_6)
result_7 = calculate_metrics(true_class_test_data_1, result_test_data_7)
result_8 = calculate_metrics(true_class_test_data_1, result_test_data_8)
result_9 = calculate_metrics(true_class_test_data_1, result_test_data_9)

fp_rate_list = [result['fp_rate'], result_1['fp_rate'], result_2['fp_rate'], result_3['fp_rate'], result_4['fp_rate'],
                result_5['fp_rate'], result_6['fp_rate'], result_7['fp_rate'], result_8['fp_rate'], result_9['fp_rate']]
tp_rate_list = [result['tp_rate'], result_1['tp_rate'], result_2['tp_rate'], result_3['tp_rate'], result_4['tp_rate'],
                result_5['tp_rate'], result_6['tp_rate'], result_7['tp_rate'], result_8['tp_rate'], result_9['tp_rate']]
print('fp', fp_rate_list)
print('tp', tp_rate_list)
plt.figure()
plt.title('Curva ROC')
plt.plot(fp_rate_list, tp_rate_list, marker=".", label="MLP")
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('Tasa de verdaderos positivos')
plt.xlabel('Tasa de falsos positivos')

plt.show()
