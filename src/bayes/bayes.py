class Bayes:
    # Atributos y la info para entrenar
    def __init__(self, attributes, classification_data,) -> None:
        self.classification_data = classification_data
        self.prob_cred_1 = 0
        self.prob_cred_0 = 0
        self.attributes = attributes
        self.frequency_attributes = []

   # Se calcula las probabilidades de las clases
    def calculate_classes_prob(self):
        sum_cred1 = 0
        sum_cred2 = 0
        data_classification_size = len(self.classification_data)
        for i in range(len(self.classification_data)):
            if (self.classification_data[i][0] == 1):
                sum_cred1 += 1
            else:
                sum_cred2 += 1
        self.prob_cred_1 = sum_cred1 / data_classification_size
        self.prob_cred_0 = sum_cred2 / data_classification_size

    # Recorremos todos los atributos y vamos haciendo un conteo de cuales se repiten

    def countall_attributes(self):
        for row in range(len(self.classification_data)):
            for col in range(1, len(self.classification_data[row])):
                self.countattribute(row, col)

    # Le sumamos 1 al atributo que se repite
    def countattribute(self, row, col):
        value = self.classification_data[row][col]
        class_value = self.classification_data[row][0]
        attributePosition = col-1
        print(attributePosition)
        self.attributes[attributePosition][class_value][value] += 1

    # Se arma un nuevo array con las frecuencias de los atributos

    def calculate_frequency(self):
        for attribute in self.attributes:
            self.frequency_attributes.append(
                {1: self.calculate(attribute[1]), 0: self.calculate(attribute[0])})

    # Se calcula la frequencia de todos los atributos de un objeto

    def calculate(self, attribute):
        attributeAux = {}
    #{1: 5, 3: 2, 4: 1}
        for key in attribute:
            attributeAux[key] = attribute[key] / sum(attribute.values())
        return attributeAux

    # Se le suma 1 a cada valor del atributo

    def laplace_correction(self):
        for attribute in self.attributes:
            for class_key in attribute:
                for key in attribute[class_key]:
                    attribute[class_key][key] += 1

    # Paso a paso de como clasificar

    def train(self):
        self.calculate_classes_prob()
        self.countall_attributes()
        self.laplace_correction()
        self.calculate_frequency()

    # Obtenemos la primer columna(clase verdadera) de una matriz

    def get_cred_ability_column(arr):
        return list(map(lambda element: element[0], arr))

    # Se comparan dos listas y se hace una comparacion para saber si es tp,tn,fp o fn,
    # Luego se realizan los calculos
    def calculate_metrics(self, arr_1, arr_2):
        tp, tn, fp, fn = 0, 0, 0, 0
        result = {}
        for i in range(len(arr_1)):
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

    # Se testea los datos de prueba y se compara con un limite si es que tiene
    def test(self, test_data, threshold=None):
        results_creditability_1 = []
        results_creditability_0 = []
        result_test_data = []
        for row in range(len(test_data)):
            prod_cred_1 = 0
            prod_cred_0 = 0
            for col in range(1, len(test_data[row])):
                attrKey = test_data[row][col]
                attrPos = col - 1
                if (prod_cred_1 == 0):
                    prod_cred_1 = self.frequency_attributes[attrPos][1][attrKey]
                    prod_cred_0 = self.frequency_attributes[attrPos][0][attrKey]
                else:
                    prod_cred_1 *= self.frequency_attributes[attrPos][1][attrKey]
                    prod_cred_0 *= self.frequency_attributes[attrPos][0][attrKey]
            prob_total = prod_cred_1 * self.prob_cred_1 + prod_cred_0*self.prob_cred_0
            results_creditability_1.append(
                prod_cred_1*self.prob_cred_1/prob_total)
            results_creditability_0.append(
                prod_cred_0*self.prob_cred_0/prob_total)
        if (threshold):
            for result in results_creditability_1:
                result_test_data.append(1 if result > threshold else 0)
        else:
            for i in range(len(results_creditability_1)):
                result_test_data.append(
                    1 if results_creditability_1[i] > results_creditability_0[i] else 0)
        return result_test_data
