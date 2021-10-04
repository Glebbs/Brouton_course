import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

data = pickle.loads(open('embeddings', "rb").read())
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open("recognizer", "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk
f = open("le", "wb")
f.write(pickle.dumps(le))
f.close()