from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

@app.route('/',methods=['GET','POST'])
def main():
  return render_template('index.html')
  
@app.route('/about',methods=['GET','POST'])
def about():
  return render_template('about.html')    	
    	  
@app.route('/data',methods=['GET','POST'])
def data():
  return render_template('data.html')
    	
@app.route('/technologies',methods=['GET','POST'])
def tech():
  return render_template('technology.html')
  
@app.route('/gold',methods=['GET','POST'])
def gold():
  return render_template('histogram_golddiff.html')
  
@app.route('/time',methods=['GET','POST'])
def time():
  return render_template('histogram_gametime.html')
  	

if __name__ == '__main__':
	#app.run()
  app.run(debug=True)
