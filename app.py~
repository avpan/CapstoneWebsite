#-*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
from bokeh.plotting import figure
from bokeh.embed import components 
from bokeh.io import gridplot,vplot
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
    df = pd.read_csv("./static/prelim_fulltimedata.csv",index_col=0)
    df = df.astype(int)
    maxtime = df['time'].max()
    results = []
    for timestamp in range(maxtime+1):
        dft = df[df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        winGoldTotal = round(dft['winGoldTotal'].mean())
        loseGoldTotal = round(dft['loseGoldTotal'].mean())
        winChampKills = round(dft['winChampKills'].mean())
        loseChampKills = round(dft['loseChampKills'].mean())
        winTowerKills = round(dft['winTowerKills'].mean())
        loseTowerKills = round(dft['loseTowerKills'].mean())
        winInhibKills = round(dft['winInhibKills'].mean())
        loseInhibKills = round(dft['loseInhibKills'].mean())
        winDragonKills = round(dft['winDragonKills'].mean())
        loseDragonKills = round(dft['loseDragonKills'].mean())
        winBaronKills = round(dft['winBaronKills'].mean())
        loseBaronKills = round(dft['loseBaronKills'].mean())
        winRiftKills = round(dft['winRiftKills'].mean())
        loseRiftKills = round(dft['loseRiftKills'].mean())
        winFirstBlood = round(dft['winFirstBlood'].mean())
        loseFirstBlood = round(dft['loseFirstBlood'].mean())
        winFirstTower = round(dft['winFirstTower'].mean())
        loseFirstTower = round(dft['loseFirstTower'].mean())
        winFirstInhib = round(dft['winFirstInhib'].mean())
        loseFirstInhib = round(dft['loseFirstInhib'].mean())
        winFirstDragon = round(dft['winFirstDragon'].mean())
        loseFirstDragon = round(dft['loseFirstDragon'].mean())
        winFirstBaron = round(dft['winFirstBaron'].mean())
        loseFirstBaron = round(dft['loseFirstBaron'].mean())
        winFirstRift = round(dft['winFirstRift'].mean())
        loseFirstRift = round(dft['loseFirstRift'].mean())
        results.append((timestamp,winGoldTotal,loseGoldTotal,winChampKills,loseChampKills, winTowerKills,loseTowerKills,winInhibKills,loseInhibKills,winDragonKills, loseDragonKills,winBaronKills, loseBaronKills,winRiftKills,loseRiftKills,   winFirstBlood,loseFirstBlood,winFirstTower,loseFirstTower,winFirstInhib,loseFirstInhib,winFirstDragon, loseFirstDragon,winFirstBaron,loseFirstBaron,winFirstRift,loseFirstRift))
        
    dft = pd.DataFrame(results,columns=['time', 
                                                                'winGoldTotal',
                                                                'loseGoldTotal', 
                                                                'winChampKills',
                                                                'loseChampKills',
                                                                'winTowerKills',
                                                                'loseTowerKills',
                                                                'winInhibKills',
                                                                'loseInhibKills',
                                                                'winDragonKills',
                                                                'loseDragonKills',
                                                                'winBaronKills',
                                                                'loseBaronKills',
                                                                'winRiftKills',
                                                                'loseRiftKills',
                                                                'winFirstBlood',
                                                                'loseFirstBlood',
                                                                'winFirstTower',
                                                                'loseFirstTower',
                                                                'winFirstInhib',
                                                                'loseFirstInhib',
                                                                'winFirstDragon',
                                                                'loseFirstDragon',
                                                                'winFirstBaron',
                                                                'loseFirstBaron',
                                                                'winFirstRift',
                                                                'loseFirstRift'])
                                               
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    hover = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    p = figure(title="Average Gold Total", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover],x_axis_label="Time (minutes)", y_axis_label = "Gold Total")
    time = dft['time']
    p1 = p.line(time,dft['winGoldTotal'],line_width=2,color='dodgerblue')
    p2 = p.circle(time,dft['winGoldTotal'],color='blue',size=5)
    legend = Legend(legends=[('Victorious Team',[p1,p2])], location=(0,-30))
    p.add_layout(legend,'left')
    
    hover2 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    s = figure(title="Average Gold Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")
    featDiff = dft['winGoldTotal']-dft['loseGoldTotal']
    s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
    s2 = s.circle(time,featDiff,line_width=2,color='indigo')
    #s.add_tools(BoxSelectTool(),hover)  
    
    vertPlot =vplot(p,s) 
    script, div = components(vertPlot)
    return render_template('timeseries.html',div=div,script=script)    
      
@app.route('/timeseries',methods=['POST'])
def timeSeries():
    df = pd.read_csv("./static/prelim_fulltimedata.csv",index_col=0)
    df = df.astype(int)
    
    maxtime = df['time'].max()
    results = []
    
    feature = request.form.getlist('features')
    feature = feature[0]
    team = request.form.getlist('team')
    for timestamp in range(maxtime+1):
        dft = df[df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        winGoldTotal = round(dft['winGoldTotal'].mean())
        loseGoldTotal = round(dft['loseGoldTotal'].mean())
        winChampKills = round(dft['winChampKills'].mean())
        loseChampKills = round(dft['loseChampKills'].mean())
        winTowerKills = round(dft['winTowerKills'].mean())
        loseTowerKills = round(dft['loseTowerKills'].mean())
        winInhibKills = round(dft['winInhibKills'].mean())
        loseInhibKills = round(dft['loseInhibKills'].mean())
        winDragonKills = round(dft['winDragonKills'].mean())
        loseDragonKills = round(dft['loseDragonKills'].mean())
        winBaronKills = round(dft['winBaronKills'].mean())
        loseBaronKills = round(dft['loseBaronKills'].mean())
        winRiftKills = round(dft['winRiftKills'].mean())
        loseRiftKills = round(dft['loseRiftKills'].mean())
        winFirstBlood = round(dft['winFirstBlood'].mean())
        loseFirstBlood = round(dft['loseFirstBlood'].mean())
        winFirstTower = round(dft['winFirstTower'].mean())
        loseFirstTower = round(dft['loseFirstTower'].mean())
        winFirstInhib = round(dft['winFirstInhib'].mean())
        loseFirstInhib = round(dft['loseFirstInhib'].mean())
        winFirstDragon = round(dft['winFirstDragon'].mean())
        loseFirstDragon = round(dft['loseFirstDragon'].mean())
        winFirstBaron = round(dft['winFirstBaron'].mean())
        loseFirstBaron = round(dft['loseFirstBaron'].mean())
        winFirstRift = round(dft['winFirstRift'].mean())
        loseFirstRift = round(dft['loseFirstRift'].mean())
        results.append((timestamp,winGoldTotal,loseGoldTotal,winChampKills,loseChampKills, winTowerKills,loseTowerKills,winInhibKills,loseInhibKills,winDragonKills, loseDragonKills,winBaronKills, loseBaronKills,winRiftKills,loseRiftKills,   winFirstBlood,loseFirstBlood,winFirstTower,loseFirstTower,winFirstInhib,loseFirstInhib,winFirstDragon, loseFirstDragon,winFirstBaron,loseFirstBaron,winFirstRift,loseFirstRift))
        
    dft = pd.DataFrame(results,columns=['time', 
                                                                'winGoldTotal',
                                                                'loseGoldTotal', 
                                                                'winChampKills',
                                                                'loseChampKills',
                                                                'winTowerKills',
                                                                'loseTowerKills',
                                                                'winInhibKills',
                                                                'loseInhibKills',
                                                                'winDragonKills',
                                                                'loseDragonKills',
                                                                'winBaronKills',
                                                                'loseBaronKills',
                                                                'winRiftKills',
                                                                'loseRiftKills',
                                                                'winFirstBlood',
                                                                'loseFirstBlood',
                                                                'winFirstTower',
                                                                'loseFirstTower',
                                                                'winFirstInhib',
                                                                'loseFirstInhib',
                                                                'winFirstDragon',
                                                                'loseFirstDragon',
                                                                'winFirstBaron',
                                                                'loseFirstBaron',
                                                                'winFirstRift',
                                                                'loseFirstRift'])
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    hover1 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    hover2 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    time = dft['time']
    
    f1,color1,color2,name = 0,'','',''
    if feature == 'goldTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winGoldTotal']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseGoldTotal']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Gold Total", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Gold Total")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Gold Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")
            featDiff = dft['winGoldTotal']-dft['loseGoldTotal']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winGoldTotal']
            f2 = dft['loseGoldTotal']
            p = figure(title="Average Gold Total", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Gold Total")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Gold Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")
            featDiff = f1 - f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------
    elif feature == 'killTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winChampKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseChampKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Champion Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Champion Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Champion Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winChampKills']-dft['loseChampKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winChampKills']
            f2 = dft['loseChampKills']
            p = figure(title="Average Champion Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Champion Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'towerTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winTowerKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseTowerKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Tower Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Tower Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Tower Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winTowerKills']-dft['loseTowerKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winTowerKills']
            f2 = dft['loseTowerKills']
            p = figure(title="Average Tower Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Tower Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Tower Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'inhibTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winInhibKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseInhibKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Inhibitor Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Inhibitor Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Inhibitor Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winInhibKills']-dft['loseInhibKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winInhibKills']
            f2 = dft['loseInhibKills']
            p = figure(title="Average Inhibitor Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Inhibitor Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Inhibitor Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'dragonTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winDragonKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseDragonKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Dragon Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Dragon Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Dragon Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winDragonKills']-dft['loseDragonKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winDragonKills']
            f2 = dft['loseDragonKills']
            p = figure(title="Average Dragon Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Dragon Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Dragon Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'baronTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winBaronKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseBaronKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Baron Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Baron Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Baron Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winBaronKills']-dft['loseBaronKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winBaronKills']
            f2 = dft['loseBaronKills']
            p = figure(title="Average Baron Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Baron Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Baron Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'riftTotal':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winRiftKills']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseRiftKills']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="Average Rift Herald Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Rift Herald Kills")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Rift Herald Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = dft['winRiftKills']-dft['loseRiftKills']
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winRiftKills']
            f2 = dft['loseRiftKills']
            p = figure(title="Average Rift Herald Kills", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "Rift Herald Kills")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average Rift Herald Kill Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Kill Difference")
            featDiff = f1-f2
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstBlood':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstBlood']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstBlood']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Blood", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Blood (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstBlood']
            f2 = dft['loseFirstBlood']
            p = figure(title="First Blood", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Blood (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstTower':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstTower']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstTower']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Tower", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Tower (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstTower']
            f2 = dft['loseFirstTower']
            p = figure(title="First Tower", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Tower (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstInhib':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstInhib']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstInhib']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Inhibitor", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Inhibitor (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstInhib']
            f2 = dft['loseFirstInhib']
            p = figure(title="First Inhibitor", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Inhibitor (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstDragon':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstDragon']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstDragon']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Dragon", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Dragon (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstDragon']
            f2 = dft['loseFirstDragon']
            p = figure(title="First Dragon", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Dragon (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstBaron':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstBaron']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstBaron']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Baron", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Baron (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstBaron']
            f2 = dft['loseFirstBaron']
            p = figure(title="First Baron", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Baron (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
    #--------------------------------------------------------------------------------    
    elif feature == 'firstRift':
        if len(team) == 1:
            f1,color1,color2,name = 0,'','',''
            if team[0] == 'victory':
                f1 = dft['winFirstRift']
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'
            else:
                f1 = dft['loseFirstRift']
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                
            p = figure(title="First Rift Herald", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Rift Herald (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color=color1)
            p2 = p.circle(time,f1,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            f1 = dft['winFirstRift']
            f2 = dft['loseFirstRift']
            p = figure(title="First Rift Herald", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "First Rift Herald (1=Yes or 0=No)")
            p1 = p.line(time,f1,line_width=2,color='dodgerblue')
            p2 = p.circle(time,f1,color='blue',size=5)
            p3 = p.line(time,f2,line_width=2,color='firebrick')
            p4 = p.circle(time,f2,color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            script, div = components(p)
            return render_template('timeseries.html',div=div,script=script)
        
 
    
if __name__ == '__main__':
    #app.run()
    app.run(debug=True)
