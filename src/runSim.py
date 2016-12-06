def warn(*args, **kwargs):
    pass
import warnings
warnings.warn=warn

import os
import pandas as pd
import numpy as np
import random
from fitHmm import HMMRunner

team_name_map = {}
team_name_map["Golden State Warriors"] = "GSW"; team_name_map["Cleveland Cavaliers"] = "CLE"
team_name_map["San Antonio Spurs"] = "SAS"; team_name_map["Toronto Raptors"] = "TOR"
team_name_map["Oklahoma City Thunder"] = "OKC"; team_name_map["Miami Heat"] = "MIA"
team_name_map["Los Angeles Clippers"] = "LAC"; team_name_map["Atlanta Hawks"] = "ATL"
team_name_map["Portland Trail Blazers"] = "POR"; team_name_map["Boston Celtics"] = "BOS"
team_name_map["Dallas Mavericks"] = "DAL"; team_name_map["Charlotte Hornets"] = "CHO"
team_name_map["Memphis Grizzlies"] = "MEM"; team_name_map["Indiana Pacers"] = "IND"
team_name_map["Houston Rockets"] = "HOU"; team_name_map["Detroit Pistons"] = "DET"
team_name_map["Utah Jazz"] = "UTA"; team_name_map["Chicago Bulls"] = "CHI"
team_name_map["Sacramento Kings"] = "SAC"; team_name_map["Washington Wizards"] = "WAS"
team_name_map["Denver Nuggets"] = "DEN"; team_name_map["Orlando Magic"] = "BOS"
team_name_map["New Orleans Pelicans"] = "NOP"; team_name_map["Milwaukee Bucks"] = "MIL"
team_name_map["Minnesota Timberwolves"] = "MIN"; team_name_map["New York Knicks"] = "NYK"
team_name_map["Phoenix Suns"] = "PHO"; team_name_map["Brooklyn Nets"] = "BRK"
team_name_map["Los Angeles Lakers"] = "LAL"; team_name_map["Philadelphia 76ers"] = "PHI"

players = pd.read_csv("../data/players/2016_adv.csv")
new_players = pd.read_csv("../data/players/2017_adv.csv")

players = players[players['G'] > 10]
players = players[players['MP'] > 5*82.]
players['Player'] = [x.split('\\')[0] for x in players['Player']]
players = players.fillna(0.)

pv_dict = {}
bpm = np.array(players['BPM']); minRatio = np.array(players['MP'])/(82.*48.)
value = bpm * minRatio
names = np.array(players['Player'])

for k in range(len(names)):
    pv_dict[names[k]] = value[k]

teams = np.unique(np.array(players['Tm']))
tv_dict = {}
for t in teams:
    tv_dict[t] = 0.

new_names = np.array([x.split('\\')[0] for x in new_players['Player']])
new_teams = np.array(new_players['Tm'])

for i in range(len(new_names)):
    if new_names[i] in pv_dict.keys():
        tv_dict[new_teams[i]] += pv_dict[new_names[i]]


runner = HMMRunner("../data/2016_adv/",'../data/players/2016_adv.csv')

newFiles = ["../data/2017_adv/" + x for x in os.listdir('../data/2017_adv/')]
newTeams = [x.split('/')[-1].split('_')[0].upper() for x in newFiles]
newFrames = [pd.read_csv(x) for x in newFiles]

teamMeans = {}
teamCovs = {}

for x in range(len(newFrames)):
    team = newTeams[x]
    currFrame = newFrames[x]
    idx = currFrame.columns.get_loc("Pace")
    currFrame['TS%'] = 100*currFrame['TS%']; currFrame['3PAr'] = 100*currFrame['3PAr']
    currFrame['eFG%'] = 100*currFrame['eFG%']; currFrame['OeFG%'] = 100*currFrame['OeFG%']

    currStats = np.array(currFrame[currFrame.columns[idx:]])
    teamMeans[team] = np.mean(currStats,axis=0)
    teamCovs[team] = np.cov(currStats,rowvar=0)

schedule_dir = "../data/2017_schedule/"
schedules = os.listdir(schedule_dir)
scheduleMap = {}

for s in schedules:
    team = (s.split('.')[0]).upper()
    opps = [x.rstrip() for x in open(schedule_dir + s).readlines()][1:]
    scheduleMap[team] = [team_name_map[x] for x in opps]


simCount = 5000
resets = 5
teamWins = {}
for team in teamMeans.keys():
    teamWins[team] = 0.0


for k in range(resets):
    for team in teamMeans.keys():
        teamStats = np.random.multivariate_normal(teamMeans[team],teamCovs[team],82)
        teamPV = np.repeat(np.array([[tv_dict[team]]]),82,axis=0)
        court = np.array([np.random.randint(2, size=82)]).T

        teamStats = np.hstack((teamStats,court))
        teamStats = np.hstack((teamStats,teamPV))
        opps = np.array(scheduleMap[team])
        oppStats = []
        for x in opps:
            oppData = np.random.multivariate_normal(teamMeans[x],teamCovs[x],1)[0]
            oppStats.append(np.append(oppData[:8],tv_dict[x]))
        oppStats = np.array(oppStats)
        totalStats = np.hstack((teamStats,oppStats))
        totalStats = runner.reader.prepro.transform(totalStats)

        probs = runner.log_reg.predict_proba(totalStats)
        if runner.polarity == 0:
            probs = probs[:,0]
        else:
            probs = probs[:,1]
    
        for _ in range(simCount):
            for p in probs:
                samp = random.uniform(0,1)
                if samp <= p:
                    teamWins[team] += 1

for t in teamMeans.keys():
    teamWins[t] /= float(simCount*resets)
    print "Team: " + str(t) + "  Wins: " + str(int(np.round(teamWins[t])))

