#-*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components 
from bokeh.io import gridplot,vplot
from bokeh.models import Legend,BoxSelectTool,HoverTool
from bokeh.charts import HeatMap, output_file, show
from bokeh.palettes import YlOrRd3,YlOrRd4,YlOrRd5
from bokeh.resources import CDN
from bokeh.embed import file_html
import requests
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import dill


app = Flask(__name__)
Bootstrap(app)
FEATURES = {"gameLength": 'Game Duration', 
                    "goldTotal": 'Total Gold',
                    'goldDiff' : 'Gold Difference',
                    'killTotal' : 'Total Champion Kills',
                    'killDiff' : 'Kill Difference',
                    'towerKills' : 'Towers Destroyed',
                    'towerDiff' : 'Tower Difference',
                    'inhibitorKills': 'Inhibitors Destroyed',
                    'inhibitorDiff':'Inhibitor Difference',
                    'dragonKills' : 'Total Dragons',
                    'dragonDiff' : 'Dragon Difference',
                    'baronKills' : 'Total Baron Nashors',
                    'baronDiff': 'Baron Nashor Difference',
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
        
        #heatmap feature importance
        feature_df = pd.read_csv("./static/feature_importances.csv",index_col=0)
        cols = list(feature_df.columns)
        cols.pop()
        TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
        hover = HoverTool(tooltips=[("(feature,time,importance)", "(@x, @y, @values)")]) 
        feature_importances = pd.melt(feature_df,id_vars=['time'],value_vars=cols,var_name='Features',value_name='Importance')
        hm = HeatMap(feature_importances,x='Features',y='time',values='Importance',stat=None,width=900,height=500,
                                palette=YlOrRd5,legend='top_right',tools=[TOOLS,hover],title='Feature Importance')
        html = file_html(hm, CDN)
        
        score_df = pd.read_csv('./static/predictionaccuracy.csv',index_col=0)
        TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
        hover1 = HoverTool(tooltips=[("time", "$index"),("(time, accuracy) (x,y)", "(@x, @y)")]) 
        p = figure(title = 'Prediction Accuracy',plot_width=900,plot_height=500,tools=[TOOLS,hover1],x_axis_label='Time(minutes)',y_axis_label='Accuracy Score')
        p.line(score_df['time'],score_df['accuracyScore'],line_width=2,color='mediumpurple')
        p.circle(score_df['time'],score_df['accuracyScore'],color='indigo',size=5)
        script, div = components(p)
        return render_template('explore.html', plot=html,script=script,div=div, features=features)    

@app.route('/distfunc',methods=['GET'])
def distFuncDefault():
    metric = FEATURES['gameLength']
    binwidth = 50
    
    df = pd.read_csv("./static/full_data.csv",index_col=0)
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
    
    df = pd.read_csv("./static/full_data.csv",index_col=0)
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
    df = pd.read_csv("./static/full_data.csv",index_col=0)
    df = df.astype(float)
    df.reset_index(inplace=True,drop=True)
    blue_team = df.iloc[::2]
    scatter = pd.scatter_matrix(blue_team[['goldDiff','towerDiff','killDiff','inhibitorDiff']], alpha=.3,figsize=(12,12),diagonal='hist');
    plt.savefig(r'./static/pictures/scatter_matrix_plot.png')
    return render_template('scatter.html')    
      
@app.route('/scattermatrix',methods=['POST'])
def scatterMatrix():
    df = pd.read_csv("./static/full_data.csv",index_col=0)
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
    df = pd.read_csv("./static/prelim_fulltimedata.csv",index_col=0,compression='zip')
    df = df.astype(int)
    winner_df = df[df['winner'] == 1]
    loser_df = df[df['winner']== 0]
    maxtime = df['time'].max()
    victory = []
    for timestamp in range(maxtime+1):
        dft = winner_df[winner_df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        goldTotal = round(dft['goldTotal'].mean())
        champKills = round(dft['champKills'].mean())
        towerKills = round(dft['towerKills'].mean())
        inhibKills = round(dft['inhibKills'].mean())
        dragonKills = round(dft['dragonKills'].mean())
        baronKills = round(dft['baronKills'].mean())
        riftKills = round(dft['riftKills'].mean())
        firstBlood = round(dft['firstBlood'].mean())
        firstTower = round(dft['firstTower'].mean())
        firstInhib = round(dft['firstInhib'].mean())
        firstDragon = round(dft['firstDragon'].mean())
        firstBaron = round(dft['firstBaron'].mean())
        firstRift = round(dft['firstRift'].mean())
        victory.append((1,timestamp,goldTotal,champKills,towerKills,inhibKills,dragonKills,baronKills,
                        riftKills,firstBlood,firstTower,firstInhib,firstDragon,firstBaron,firstRift))
    
    defeat = []
    for timestamp in range(maxtime+1):
        dft = loser_df[loser_df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        goldTotal = round(dft['goldTotal'].mean())
        champKills = round(dft['champKills'].mean())
        towerKills = round(dft['towerKills'].mean())
        inhibKills = round(dft['inhibKills'].mean())
        dragonKills = round(dft['dragonKills'].mean())
        baronKills = round(dft['baronKills'].mean())
        riftKills = round(dft['riftKills'].mean())
        firstBlood = round(dft['firstBlood'].mean())
        firstTower = round(dft['firstTower'].mean())
        firstInhib = round(dft['firstInhib'].mean())
        firstDragon = round(dft['firstDragon'].mean())
        firstBaron = round(dft['firstBaron'].mean())
        firstRift = round(dft['firstRift'].mean())
        defeat.append((0,timestamp,goldTotal,champKills,towerKills,inhibKills,dragonKills,baronKills,
                        riftKills,firstBlood,firstTower,firstInhib,firstDragon,firstBaron,firstRift))

    dft1 = pd.DataFrame(victory,columns=['winner','time','goldTotal', 'killTotal','towerKills','inhibitorKills','dragonKills','baronKills',
                                       'riftHeraldKills','firstBlood','firstTower','firstInhibitor','firstDragon','firstBaron','firstRiftHerald'])
    dft2 = pd.DataFrame(defeat,columns=['winner','time','goldTotal', 'killTotal','towerKills','inhibitorKills','dragonKills','baronKills',
                                           'riftHeraldKills','firstBlood','firstTower','firstInhibitor','firstDragon','firstBaron','firstRiftHerald'])

    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    hover = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    p = figure(title="Average Gold Total", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover],x_axis_label="Time (minutes)", y_axis_label = "Gold Total")
    time = dft1['time']
    p1 = p.line(time,dft1['goldTotal'],line_width=2,color='dodgerblue')
    p2 = p.circle(time,dft1['goldTotal'],color='blue',size=5)
    p3 = p.line(time,dft2['goldTotal'],line_width=2,color='firebrick')
    p4 = p.circle(time,dft2['goldTotal'],color='red',size=5)
    legend = Legend(legends=[('Victorious Team',[p1,p2]),('Defeat Team',[p3,p4])], location=(0,-30))
    p.add_layout(legend,'left')
    
    hover2 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    s = figure(title="Average Gold Difference", plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "Gold Difference")
    featDiff = dft1['goldTotal']-dft2['goldTotal']
    s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
    s2 = s.circle(time,featDiff,line_width=2,color='indigo')
    vertPlot =vplot(p,s) 
    script, div = components(vertPlot)
    return render_template('timeseries.html',div=div,script=script)    
      
@app.route('/timeseries',methods=['POST'])
def timeSeries():
    df = pd.read_csv("./static/fulltimedata.csv.zip",index_col=0,compression='zip')
    df = df.astype(int)
    winner_df = df[df['winner'] == 1]
    loser_df = df[df['winner']==0]
    maxtime = df['time'].max()
    victory, defeat = [],[]
    
    feature = request.form.getlist('features')
    feature = feature[0]
    team = request.form.getlist('team')
    for timestamp in range(maxtime+1):
        dft = winner_df[winner_df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        goldTotal = round(dft['goldTotal'].mean())
        champKills = round(dft['champKills'].mean())
        towerKills = round(dft['towerKills'].mean())
        inhibKills = round(dft['inhibKills'].mean())
        dragonKills = round(dft['dragonKills'].mean())
        baronKills = round(dft['baronKills'].mean())
        riftKills = round(dft['riftKills'].mean())
        firstBlood = round(dft['firstBlood'].mean())
        firstTower = round(dft['firstTower'].mean())
        firstInhib = round(dft['firstInhib'].mean())
        firstDragon = round(dft['firstDragon'].mean())
        firstBaron = round(dft['firstBaron'].mean())
        firstRift = round(dft['firstRift'].mean())
        victory.append((1,timestamp,goldTotal,champKills,towerKills,inhibKills,dragonKills,baronKills,
                        riftKills,firstBlood,firstTower,firstInhib,firstDragon,firstBaron,firstRift))

    for timestamp in range(maxtime+1):
        dft = loser_df[loser_df['time']==timestamp]
        if dft.shape[0] <= 50:
            continue
        goldTotal = round(dft['goldTotal'].mean())
        champKills = round(dft['champKills'].mean())
        towerKills = round(dft['towerKills'].mean())
        inhibKills = round(dft['inhibKills'].mean())
        dragonKills = round(dft['dragonKills'].mean())
        baronKills = round(dft['baronKills'].mean())
        riftKills = round(dft['riftKills'].mean())
        firstBlood = round(dft['firstBlood'].mean())
        firstTower = round(dft['firstTower'].mean())
        firstInhib = round(dft['firstInhib'].mean())
        firstDragon = round(dft['firstDragon'].mean())
        firstBaron = round(dft['firstBaron'].mean())
        firstRift = round(dft['firstRift'].mean())
        defeat.append((0,timestamp,goldTotal,champKills,towerKills,inhibKills,dragonKills,baronKills,
                        riftKills,firstBlood,firstTower,firstInhib,firstDragon,firstBaron,firstRift))
                                
    dft1 = pd.DataFrame(victory,columns=['winner','time','goldTotal', 'killTotal','towerKills','inhibitorKills','dragonKills','baronKills',
                                       'riftHeraldKills','firstBlood','firstTower','firstInhibitor','firstDragon','firstBaron','firstRiftHerald'])
    dft2 = pd.DataFrame(defeat,columns=['winner','time','goldTotal', 'killTotal','towerKills','inhibitorKills','dragonKills','baronKills',
                                           'riftHeraldKills','firstBlood','firstTower','firstInhibitor','firstDragon','firstBaron','firstRiftHerald'])

                                           
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    hover1 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    hover2 = HoverTool(tooltips=[("time", "$index"),("(time, value) (x,y)", "(@x, @y)")]) 
    time = dft1['time']
    
    f1,color1,color2,name = 0,'','',''
    if 'first' not in feature:
        if len(team) == 1:
            dft,color1,color2,name =[],'','',''
            if team[0] == 'victory':
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'       
                dft = dft1[feature]     
            else:
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                dft = dft2[feature]
            p = figure(title="Average %s" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "%s" % FEATURES[feature])    
            p1 = p.line(time,dft,line_width=2,color=color1)
            p2 = p.circle(time,dft,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')  
            
            s = figure(title="Average %s Difference" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "%s Difference" % FEATURES[feature])
            featDiff = dft1[feature]-dft2[feature]
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)            
        else:
            p = figure(title="Average %s" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "%s" % FEATURES[feature])
            p1 = p.line(time,dft1[feature],line_width=2,color='dodgerblue')
            p2 = p.circle(time,dft1[feature],color='blue',size=5)
            p3 = p.line(time,dft2[feature],line_width=2,color='firebrick')
            p4 = p.circle(time,dft2[feature],color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')    
            
            s = figure(title="Average %s Difference" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover2],x_axis_label="Time (minutes)", y_axis_label = "%s Difference" % FEATURES[feature])
            featDiff = dft1[feature]-dft2[feature]
            s1 = s.line(time,featDiff,line_width=2,color='mediumpurple')
            s2 = s.circle(time,featDiff,line_width=2,color='indigo')
            
            vertPlot =vplot(p,s) 
            script, div = components(vertPlot)
            return render_template('timeseries.html',div=div,script=script)
    else:
        if len(team) == 1:
            dft,color1,color2,name =[],'','',''
            if team[0] == 'victory':
                color1 = 'dodgerblue'
                color2 = 'blue'
                name = 'Victorious Team'       
                dft = dft1[feature]     
            else:
                color1 = 'firebrick'
                color2 = 'red'
                name = 'Defeat Team'
                dft = dft2[feature]
                
            p = figure(title="Average %s" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "%s (1=True, 0=False)" % FEATURES[feature])    
            p1 = p.line(time,dft,line_width=2,color=color1)
            p2 = p.circle(time,dft,color=color2,size=5)
            legend = Legend(legends=[(name,[p1,p2])], location=(0,-30))
            p.add_layout(legend,'left')  
            script,div = components(p)
            return render_template('timeseries.html',div=div,script=script)
        else:  
            p = figure(title="Average %s" % FEATURES[feature], plot_width=900,plot_height=300,tools=[TOOLS,BoxSelectTool(),hover1],x_axis_label="Time (minutes)", y_axis_label = "%s (1=True, 0=False)" % FEATURES[feature])
            p1 = p.line(time,dft1[feature],line_width=2,color='dodgerblue')
            p2 = p.circle(time,dft1[feature],color='blue',size=5)
            p3 = p.line(time,dft2[feature],line_width=2,color='firebrick')
            p4 = p.circle(time,dft2[feature],color='red',size=5)
            legend = Legend(legends=[('Victorious Team',[p1,p2]), ('Defeat Team',[p3,p4])], location=(0,-30))
            p.add_layout(legend,'left')  
            script,div = components(p)
            return render_template('timeseries.html',div=div,script=script)    
            
@app.route('/predictor',methods=['GET'])
def predictor():     
    error = ''                     
    return render_template('predictor.html')

@app.route('/predictor',methods=['POST'])
def predictorPost():        
    goldTotal = request.form.getlist('goldTotal')
    champKills = request.form.getlist('champKills')
    towerKills = request.form.getlist('towerKills')
    inhibKills = request.form.getlist('inhibKills')
    dragonKills = request.form.getlist('dragonKills')
    baronKills = request.form.getlist('baronKills')
    predictTeam = request.form.getlist('predictTeam')[0]
    #firstBlood = request.form.getlist('firstBlood')
    #firstTower = request.form.getlist('firstTower')
    time = int(request.form.getlist('time')[0])
    #print request.form.getlist('firstBlood'), request.form.getlist('firstTower')
    if predictTeam == 'blue':
        team1 = 0
        team2 = 1
    else:
        team1 = 1
        team2 = 0
        
    goldDiff = int(goldTotal[team1]) - int(goldTotal[team2])
    killDiff = int(champKills[team1]) - int(champKills[team2])
    towerDiff = int(towerKills[team1]) - int(towerKills[team2])
    inhibDiff = int(inhibKills[team1]) - int(inhibKills[team2])
    dragonDiff = int(dragonKills[team1]) - int(dragonKills[team2])
    baronDiff = int(baronKills[team1]) - int(baronKills[team2])
    
    features =[(goldTotal[team1],goldDiff,champKills[team1],killDiff,towerKills[team1],towerDiff,inhibKills[team1],inhibDiff,dragonKills[team1],dragonDiff,baronKills[team1],baronDiff)]
    features_df = pd.DataFrame(features,columns=['goldTotal','goldDiff','champKills','killDiff','towerKills','towerDiff','inhibKills','inhibDiff','dragonKills', 'dragonDiff','baronKills','baronDiff'])
    score_df = pd.read_csv('./static/predictionaccuracy.csv',index_col=0) 
    error = ''
    if time > score_df.shape[0]:
        return render_template('predictor.html',error="Game is too long for reliable prediction. It is up to the player's skills now.")
    else:
        with open('./static/models/randomforestmodel_esports_%d.pkl'%time,'rb') as infile:
            forest = dill.load(infile)  
        win_percent = round(score_df.loc[time,'accuracyScore']*100,2)
        winner = forest.predict(features_df)
        if winner == 0:
            win_percent = 100-win_percent
            
        print "Team has a %f of winning" % win_percent
        return render_template('predictor.html',win_percent=win_percent,error=error)
if __name__ == '__main__':
    #app.run()
    app.run(debug=True)
