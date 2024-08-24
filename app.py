from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__,template_folder='template1',static_folder='static')

svm_model=pickle.load(open('svm.pkl','rb'))
ann_model=pickle.load(open('ann.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('prac.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    print(request.form)
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    ann_pred=ann_model.predict_proba(final)
    svm_pred=svm_model.predict_proba(final)
    a= np.mean(ann_pred)
    b= np.mean(svm_pred)
    w1 = a/(a+b)
    w2 = b/(a+b)
    combined_proba = w1 * ann_pred + w2 * svm_pred
    final_pred=np.argmax(combined_proba,axis=1)
    output = '{0:.{1}f}'.format(final_pred[0], 2)
    if output>str(0.5):
        return render_template('result1.html',pred='Negative')
    else:
        return render_template('result1.html',pred='Positive')

if __name__=='__main__':
    app.run()
