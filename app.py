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
    features = [('goldDiff', 0.97447258779181267),
    ('towerDiff', 0.97252165862046158),
    ('killDiff', 0.92064016930097214),
    ('inhibitorDiff', 0.91908603928311616),
    ('towerKills', 0.90946365981085908),
    ('firstInhibitor', 0.88066265458633686),
    ('inhibitorKills', 0.79101911249255996),
    ('dragonDiff', 0.7774948746776007),
    ('dragonKills', 0.75332319291052174),
    ('killTotal', 0.73473976588849943),
    ('baronDiff', 0.71380861054163081),
    ('baronKills', 0.71172541498578135),
    ('firstTower', 0.70054890549566828),
    ('firstBaron', 0.69922624165068448),
    ('firstDragon', 0.67002843727266714),
    ('goldTotal', 0.66199325441439061),
    ('firstBlood', 0.58951127570927853),
    ('firstRiftHerald', 0.54807883076516106),
    ('riftHeraldKills', 0.54807883076516106),
    ('gameLength', 0.5)]
    return render_template('explore.html', features = features)    

@app.route('/game',methods=['GET','POST'])
def howto():
  return render_template('howto.html')      	

if __name__ == '__main__':
    app.run()
  #app.run(debug=True)
