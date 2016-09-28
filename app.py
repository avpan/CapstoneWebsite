from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
from bokeh.plotting import figure
from bokeh.embed import components 
from bokeh.io import gridplot
import requests
import pandas as pd
import csv
import datetime
import numpy as py 

app = Flask(__name__)
Bootstrap(app)

@app.route('/',methods=['GET','POST'])
def main():
  return render_template('index.html')
  
@app.route('/about',methods=['GET','POST'])
def about():
  return render_template('about.html')    
  
@app.route('/contact',methods=['GET','POST'])
def contact():
  return render_template('contact.html')    		
    	  
@app.route('/getdata',methods=['GET','POST'])
def data():
  return render_template('data.html')
    	
@app.route('/technologies',methods=['GET','POST'])
def tech():
  return render_template('technology.html')
  
@app.route('/explore',methods=['GET','POST'])
def explore():
  return render_template('explore.html')    

@app.route('/game',methods=['GET','POST'])
def howto():
  return render_template('howto.html')      	

if __name__ == '__main__':
    app.run()
  #app.run(debug=True)
