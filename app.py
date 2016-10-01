#-*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
from bokeh.plotting import figure
from bokeh.embed import components 
from bokeh.io import gridplot
from bokeh.models import Legend,BoxSelectTool,HoverTool
import requests
import pandas as pd
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
Bootstrap(app)
FEATURES = {"gameLength": 'Game Duration', 
                    "goldTotal": 'Total Gold',
                    'goldDiff' : 'Gold Difference',
                    'killTotal' : 'Total Kills',
                    'killDiff' : 'Kill Difference',
                    'towerKills' : 'Towers Destroyed',
                    'towerDiff' : 'Tower Difference',
                    'inhibitorKills': 'Inhibitors Destroyed',
                    'inhibitorDiff':'Inhibitor Difference',
                    'dragonKills' : 'Total Dragons',
                    'dragonDiff' : 'Dragon Difference',
                    'baronKills' : 'Total Barons',
                    'riftHeraldKills' : 'Rift Herald Kills',
                    'firstBaron' : 'First Baron',
                    'firstDragon': 'First Dragon',
                    'firstRiftHerald':'First Rift Herald',
                    'firstBlood': 'First Blood',
                    'firstTower': 'First Tower',
                    'firstInhibitor':'First Inhibitor'}

@app.route('/',methods=['GET','POST'])
def main():
  return render_template('index.html')
  
@app.route('/about',methods=['GET','POST'])
def about():
  return render_template('about.html')    
  
@app.route('/game',methods=['GET','POST'])
def howto():
  return render_template('howto.html')    
    
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
    if request.method == 'GET':
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
    #else:
        #return redirect('/distfunc')  	
  
@app.route('/distfunc',methods=['GET'])
def distFuncDefault():
    metric = FEATURES['gameLength']
    binwidth = 50
    
    df = pd.read_csv("./static/prelim_data.csv",index_col=0)
    df = df.astype(float)
    df.reset_index(inplace=True,drop=True)
    
    #plotting in bokeh
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    
    blue_team = df.iloc[::2]
    feature_data = blue_team[['gameLength']]
    mean = feature_data.mean()
    std = feature_data.std()
    p = figure(title="%s Distribution(μ=%.1f, σ=%.1f)"%(metric,mean,std),plot_width=900, plot_height=600, tools=TOOLS, x_axis_label = '%s' % metric, y_axis_label = '# Games')
    hist,edges = np.histogram(feature_data,bins=binwidth)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="dodgerblue", line_color="black", legend = 'Victorious Team')
    script, div = components(p)
    return render_template('distribution.html', script=script, div=div)
    
@app.route('/distfunc',methods=['POST'])
def distFunc():
    if request.method == 'POST':
        redirect('/distfunc')   
    feature = request.form.getlist('dist_feature')
    team = request.form.getlist('team')   
    bin = request.form.getlist('binwidth')
    metric = FEATURES[feature[0]]
    binwidth = int(bin[0])
    
    df = pd.read_csv("./static/prelim_data.csv",index_col=0)
    df = df.astype(float)
    df.reset_index(inplace=True,drop=True)
    hist,edges = [],[]
    #plotting in bokeh
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    if len(team) == 1:
        if team[0] == 'victory':
            blue_team = df.iloc[::2]
            feature_data = blue_team[[feature[0]]]
            mean = feature_data.mean()
            std = feature_data.std()
            p = figure(title="%s Distribution(μ=%.1f, σ=%.1f)"%(metric,mean,std),plot_width=900, plot_height=600, tools=TOOLS, x_axis_label = '%s' % metric, y_axis_label = '# Games')
            hist,edges = np.histogram(feature_data,bins=binwidth)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="dodgerblue", line_color="black", legend = 'Victorious Team')
            
        else:
            red_team = df.iloc[1::2]
            feature_data = red_team[[feature[0]]]
            mean = feature_data.mean()
            std = feature_data.std()
            p = figure(title="%s Distribution(μ=%.1f, σ=%.1f)"%(metric,mean,std),plot_width=900, plot_height=600, tools=TOOLS, x_axis_label = '%s' % metric, y_axis_label = '# Games')
            hist,edges = np.histogram(feature_data,bins=binwidth)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="firebrick", line_color="black", legend = 'Defeated Team')
    elif len(team) == 2:
        blue_team = df.iloc[::2]
        red_team = df.iloc[1::2]
        victory = blue_team[[feature[0]]]
        defeat = red_team[[feature[0]]]
        victory_mean = victory.mean()
        victory_std = victory.std()
        defeat_mean = defeat.mean()
        defeat_std = defeat.std()
        p = figure(title="%s Distribution" % metric, plot_width=900,plot_height=600,tools=TOOLS,x_axis_label="%s" % metric, y_axis_label = "# Games")
        hist1, edges1 = np.histogram(victory, bins=binwidth)
        hist2, edges2 = np.histogram(defeat, bins=binwidth)
        p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:], fill_color="dodgerblue", line_color="black", legend = 'Victorious Team')
        p.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:], fill_alpha=.8, fill_color="firebrick", line_color="black", legend = 'Defeated Team')
    script, div = components(p)
    return render_template('distribution.html', script=script, div=div)

@app.route('/scattermatrix',methods=['GET'])
def scatterDefault():
    df = pd.read_csv("./static/prelim_data.csv",index_col=0)
    df = df.astype(float)
    df.reset_index(inplace=True,drop=True)
    blue_team = df.iloc[::2]
    scatter = pd.scatter_matrix(blue_team[['goldDiff','towerDiff','killDiff','inhibitorDiff']], alpha=.3,figsize=(12,12),diagonal='hist');
    plt.savefig(r'./static/pictures/scatter_matrix_plot.png')
    return render_template('scatter.html')    
      
@app.route('/scattermatrix',methods=['POST'])
def scatterMatrix():
    df = pd.read_csv("./static/prelim_data.csv",index_col=0)
    df = df.astype(float)
    df.reset_index(inplace=True,drop=True)
    
    team = request.form.getlist('team')
    features = request.form.getlist('features')
    diagonal = request.form.getlist('diagonal')
    diagonal = diagonal[0]
    size = len(features)+10
    if len(team) == 1:
        if team[0] == 'victory':
            blue_team = df.iloc[::2]
            scatter = pd.scatter_matrix(blue_team[features], alpha=.3,figsize=(size,size),diagonal=diagonal);
            plt.savefig('./static/pictures/scatter_matrix_plot.png')
        else:
            red_team = df.iloc[1::2]
            scatter = pd.scatter_matrix(red_team[features], alpha=.3,figsize=(size,size),diagonal=diagonal);
            plt.savefig('./static/pictures/scatter_matrix_plot.png')
    return render_template('scatter.html')   

@app.route('/timeseries',methods=['GET'])
def timeSeriesDefault():    
    df = pd.read_csv("./static/prelim_timedata.csv",index_col=0)
    maxtime = df['time'].max()
    results = []
    for timestamp in range(maxtime+1):
        dft = df[df['time']==timestamp]
        vicTotal_mean = dft['vicGoldTotal'].mean()
        vicTotal_std = dft['vicGoldTotal'].std()
        defTotal_mean = dft['defGoldTotal'].mean()
        defTotal_std = dft['vicGoldTotal'].std()
        goldDiff_mean = dft['goldDiff'].mean()
        goldDiff_std = dft['goldDiff'].std()
        results.append((timestamp,vicTotal_mean,vicTotal_std,defTotal_mean,defTotal_std,goldDiff_mean,goldDiff_std))
    dft = pd.DataFrame(results,columns=['time','vicTotalMean','vicTotalStd','defTotalMean','defTotalStd','goldDiffMean','goldDiffStd'])
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    p = figure(title="Time Series Average for Gold Difference", plot_width=900,plot_height=600,tools=TOOLS,x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")
    hover = HoverTool(
            tooltips=[
                ("time", "$index"),
                ("(time, value) (x,y)", "(@x, @y)")
                ]) 
    p.add_tools(BoxSelectTool(),hover)
    data_mean = dft['goldDiffMean']
    time = dft['time']
    p1 = p.line(time,data_mean,line_width=2,color='mediumpurple')
    p2 = p.circle(time,data_mean,color='indigo',size=5)
    legend = Legend(legends=[('Gold Diff of Teams',[p1,p2])], location=(0,-30))
    p.add_layout(legend,'left')
    script, div = components(p)
    return render_template('timeseries.html',div=div,script=script)    
      
@app.route('/timeseries',methods=['POST'])
def timeSeries():
    df = pd.read_csv("./static/prelim_timedata.csv",index_col=0)
    maxtime = df['time'].max()
    results = []
    feature = request.form.getlist('features')
    feature = feature[0]
    team = request.form.getlist('team')
    for timestamp in range(maxtime+1):
        dft = df[df['time']==timestamp]
        vicTotal_mean = dft['vicGoldTotal'].mean()
        vicTotal_std = dft['vicGoldTotal'].std()
        defTotal_mean = dft['defGoldTotal'].mean()
        defTotal_std = dft['vicGoldTotal'].std()
        goldDiff_mean = dft['goldDiff'].mean()
        goldDiff_std = dft['goldDiff'].std()
        results.append((timestamp,vicTotal_mean,vicTotal_std,defTotal_mean,defTotal_std,goldDiff_mean,goldDiff_std))
    dft = pd.DataFrame(results,columns=['time','vicTotalMean','vicTotalStd','defTotalMean','defTotalStd','goldDiffMean','goldDiffStd'])
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    time = dft['time']
    hover = HoverTool(
            tooltips=[
                ("time", "$index"),
                ("(time, value) (x,y)", "(@x, @y)")
                ]) 
    if feature == 'goldDiff':
        p = figure(title="Time Series Average for Gold Difference", plot_width=900,plot_height=600,tools=TOOLS,x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")        
        data_mean = dft['goldDiffMean']
        p1 = p.line(time,data_mean,line_width=2,color='mediumpurple')
        p2 = p.circle(time,data_mean,color='indigo',size=5)
        legend = Legend(legends=[('Gold Diff of Teams',[p1,p2])], location=(0,-30))
    else:
        p = figure(title="Time Series Average for Total Gold Earned", plot_width=900,plot_height=600,tools=TOOLS,x_axis_label="Time (minutes)", y_axis_label = "Total Gold Earned")
        if len(team) == 1:
            if team[0] == 'victory':
                data_mean = dft['vicTotalMean']
                p1 = p.line(time,data_mean,line_width=2,color='dodgerblue')
                p2 = p.circle(time,data_mean,color='blue',size=5)
                legend = Legend(legends=[('Victorious Team',[p1,p2])], location=(0,-30))
            else:
                data_mean = dft['defTotalMean']
                p1 = p.line(time,data_mean,line_width=2,color='firebrick')
                p2 = p.circle(time,data_mean,color='red',size=5)
                legend = Legend(legends=[('Defeated Team',[p1,p2])], location=(0,-30))
        else:
            blue_team = dft['vicTotalMean']
            red_team = dft['defTotalMean']
            p1 = p.line(time,blue_team,line_width=2,color='dodgerblue')
            p2 = p.circle(time,blue_team,color='blue',size=5)
            p3 = p.line(time,red_team,line_width=2,color='firebrick')
            p4 = p.circle(time,red_team,line_width=2,color='red')
            legend = Legend(legends=[('Victorious Team', [p1,p2]),('Defeated Team',[p3,p4])], location=(0,-30))
    p.add_tools(BoxSelectTool(),hover)    
    p.add_layout(legend,'left')    
    script, div = components(p)
    return render_template('timeseries.html',div=div,script=script)    
    
if __name__ == '__main__':
    #app.run()
    app.run(debug=True)
