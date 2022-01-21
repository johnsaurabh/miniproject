import numpy as np 
import tensorflow as tf
from flask import Flask, request, jsonify, render_template 
from keras.models import  load_model
from preprocess import vectorize_stories, tokenize, story_maxlen, query_maxlen, idx_word, word_idx 

app = Flask(__name__) 

#model = load_model('dnm_model.h5') 

global graph
graph=tf.compat.v1.get_default_graph()


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():

  
    '''
    For rendering results on HTML GUI
    '''
    with graph.as_default():
        model = load_model('qa1_single_model.h5')  
        
        user_story_inp = tokenize(request.form.get('story') ) 
        user_query_inp = tokenize(request.form.get('question'))
      
        user_story, user_query, user_ans = vectorize_stories([[user_story_inp, user_query_inp, '.']], word_idx, story_maxlen, query_maxlen)
        user_prediction = model.predict([user_story, user_query])
        user_prediction = idx_word[np.argmax(user_prediction)]
        # return jsonify(user_prediction)
        return render_template('index.html', prediction_text='Answer is {}'.format(user_prediction))
    
 

if __name__ == "__main__":
    app.run(debug=False)
    