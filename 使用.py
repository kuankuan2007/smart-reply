import pickle,json,jieba
with open("model.pkl","rb") as m:
    modal=pickle.load(m)
with open("ai.pkl","rb") as m:
    clf=pickle.load(m)
while True:
    ans=input()
    message="|".join(jieba.cut(ans))
    aiVare=modal.transform([message])
    print(aiVare.toarray())
    retsult=clf.predict(aiVare)[0]
    print(retsult)