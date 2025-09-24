from MODELS.perceptron import Perceptron
from datasets import makeLinearDS, makeNonLinearDS, loadTitanicDS
from MODELS.baseline import RandomBaseline
from MODELS.adaline import AdalineGD


x_lin, y_lin = makeLinearDS()
x_nonlin, y_nonlin = makeNonLinearDS()
x_train, x_test, y_train, y_test = loadTitanicDS()


 # -------- Linear Separable Dataset --------


baseline = RandomBaseline()
baseline.fit(x_lin, y_lin)
print("Random baseline accuracy: ", (baseline.predict(x_lin) == y_lin).mean())


perceptron = Perceptron()
perceptron.fit(x_lin, y_lin)
print("Linear Perceptron accuracy: ", (perceptron.predict(x_lin) == y_lin).mean())


adaline = AdalineGD(eta=0.01, n_iter=1000)
adaline.fit(x_lin, y_lin)
print("Linear Adaline accuracy: ", (adaline.predict(x_lin) == y_lin).mean())


# -------- Non-Linear Separable Dataset --------


baselineNL = RandomBaseline()
baselineNL.fit(x_nonlin, y_nonlin)
print("Non-Linear Random baseline accuracy: ", (baselineNL.predict(x_nonlin) == y_nonlin).mean())

perceptronNL = Perceptron()
perceptronNL.fit(x_nonlin, y_nonlin)
print("Non-Linear Perceptron accuracy: ", (perceptronNL.predict(x_nonlin) == y_nonlin).mean())


adalineNL = AdalineGD(eta=0.01, n_iter=1000)
adalineNL.fit(x_nonlin, y_nonlin)
print("Non-Linear Adaline accuracy: ", (adalineNL.predict(x_nonlin) == y_nonlin).mean())


# -------- Titanic Dataset --------


baselineTitanic = RandomBaseline()
baselineTitanic.fit(x_train, y_train)
print("Random baseline accuracy: ", (baselineTitanic.predict(x_test) == y_test).mean())

perceptronTitanic = Perceptron()
perceptronTitanic.fit(x_train, y_train)
print("Titanic DS accuracy: ", (perceptronTitanic.predict(x_test) == y_test).mean())

adalineTitanic = AdalineGD(eta=0.01, n_iter=1000)
adalineTitanic.fit(x_train, y_train)
print("Titanic Adaline accuracy: ", (adalineTitanic.predict(x_test) == y_test).mean())