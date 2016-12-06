import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataReader:
    def __init__(self,data_dir,player_file):
        self.players = pd.read_csv(player_file)
        self.files = [data_dir + x for x in os.listdir(data_dir)]
        self.teams = [x.split('/')[-1].split('_')[0].upper() for x in self.files]
        self.frames = [pd.read_csv(x) for x in self.files]
        self.teamMeans = {}; self.teamCovs = {}
        self.featureNames = []
        self.preprocess_players()
        self.prepro = StandardScaler()
        self.preprocess()
    
    def preprocess_players(self):
        #getting rid of transient players
        #self.players = self.players[self.players['G'] > 35]
        self.players = self.players[self.players['MP'] > 7.*35.]

        #reformatting player names
        self.players['Player'] = [x.split('\\')[0] for x in self.players['Player']]

        #fill missing values
        self.players = self.players.fillna(0.)

        #getting weighted value over replacement player
        self.rosterMap = {}
        for team in self.teams:
            players = self.players[self.players['Tm']==team]
            minRatio = np.array(players['MP'])/(82.*48.)
            winShares = np.array(players['BPM'])
            self.rosterMap[team] = np.array([[np.dot(winShares,minRatio)]])
    
    
    def preprocess(self):
        self.records = []
        self.stats = []
        for i in range(len(self.teams)):
            dFrame = self.frames[i]
            idx = dFrame.columns.get_loc("Pace")
            dFrame['W/L'].replace(to_replace=dict(W=1,L=0), inplace=True)
            dFrame['Court'].fillna(1,inplace=True)
            dFrame['Court'].replace(to_replace='@',value=0,inplace=True)

            dFrame['TS%'] = 100*dFrame['TS%']; dFrame['3PAr'] = 100*dFrame['3PAr']
            dFrame['eFG%'] = 100*dFrame['eFG%']; dFrame['OeFG%'] = 100*dFrame['OeFG%']

            teamStats = np.array(dFrame[dFrame.columns[idx:]])
            teamMean = np.mean(teamStats,axis=0)
            teamCov = np.cov(teamStats,rowvar=0)
            #getting player stats as prior for model
           
            self.teamMeans[self.teams[i]] = teamMean
            self.teamCovs[self.teams[i]] = teamCov

        for i in range(len(self.frames)):
            dFrame = self.frames[i]
            homeTeam = self.teams[i]
            homePV = np.repeat(self.rosterMap[homeTeam],82,axis=0)
            
          
            idx = dFrame.columns.get_loc("Pace")
            self.featureNames = list(dFrame.columns[idx:])
            teamStats = np.array(dFrame[dFrame.columns[idx:]])

            self.featureNames.append('HomeCourt')
            teamStats = np.hstack((teamStats,np.array([dFrame['Court']]).T))
            
            self.featureNames.append("TeamValue")
            teamStats = np.hstack((teamStats,homePV))

            results = []
            oppTeams = np.array(dFrame['Opp'])
            seriesCount = {}
            for k in np.unique(oppTeams):
                seriesCount[k] = 0

            for t in oppTeams:
                tid = self.teams.index(t)
                oppPV = self.rosterMap[t][0][0]
                f = self.frames[tid]; f = f[f['Opp'] == homeTeam]

                idx = f.columns.get_loc("Pace"); endIdx = f.columns.get_loc("eFG%")
                for k in f.columns[idx:endIdx]:
                    self.featureNames.append("Opp"+k)

                oppData = np.array(f[f.columns[idx:endIdx]])

                self.featureNames.append("oppValue")
                resData = np.append(oppData[seriesCount[t]],oppPV)
                results.append(resData)
                seriesCount[t] += 1
            
            oppStats = np.array(results)
            assert sum(seriesCount.values()) == 82
            assert oppStats.shape[0] == 82

            totalStats = np.hstack((teamStats,oppStats))
            self.stats.append(totalStats)
            self.records.append(np.array(dFrame['W/L']))

    def get_data(self):
        y = self.records[0]
        X = self.stats[0]
        lengths = [len(X)]
        for i in range(1,len(self.records)):
            X = np.vstack((X,self.stats[i]))
            y = np.concatenate((y,self.records[i]))
            lengths.append(len(self.stats[i]))
        
        #preprocess the columns for numerical stability
        X = self.prepro.fit_transform(X)
        return X,y,lengths

