from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd
from sklearn import tree
import pickle,sys
file=open("output","w",encoding="utf-8")
# a=sys.stdout
# sys.stdout=file
df = pd.DataFrame(pd.read_excel('talk20230118.xlsx'))
df=df.fillna(value="对不起,我没有理解到你的意思")
df=df[:600]
print(df)
x=df["input"].to_list()
print(x)
cutX = ["|".join(jieba.cut(i)) for i in x]
print(cutX)

vectorizer = CountVectorizer(stop_words = ['好的','是的'],decode_error="replace")
modal=vectorizer.fit(cutX)
intValue=modal.transform(cutX)
num=0
for i in intValue:
    if "with 0" in str(list(i)):
        print(num,df["input"][num],df["output"][num])
    num=num+1
print(modal.vocabulary_)
f = tree.DecisionTreeClassifier()
y=df["output"]
# sys.stdout=a
# file.close()
print("training")
clf = f.fit(intValue,y)
with open("ai.pkl","wb") as m:
    pickle.dump(clf,m)
with open("model.pkl","wb") as m:
    pickle.dump(modal,m)
print("OK")
while True:
    s=input()
    s="|".join(jieba.cut(s))
    intValue=modal.transform([s])
    print(intValue.toarray())
    e=clf.predict(intValue)
    print(e)
