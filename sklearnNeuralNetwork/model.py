import csv
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler()

X = []
y = []

path = 'BreastCancerDataset.csv'

with open(path) as csvfile:
    data = list(csv.DictReader(csvfile))

#print "Features"
#print data[0].keys()

for x in data:
	X.append([x["area_se"],x["smoothness_se"],x["compactness_se"]])
	y.append(x["diagnosis"])

for x in xrange(1,10):
	print X[x]

scaler.fit(X)
X = scaler.transform(X)

for x in xrange(1,10):
	print X[x]

#mlp = MLPClassifier(hidden_layer_sizes=(3))
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[3], max_iter=2000, activation='logistic')
mlp.fit(X,y)
predictions = mlp.predict([[165, 0.006, 0.04]])	
print predictions 
			