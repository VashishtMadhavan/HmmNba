def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from hmmlearn import hmm
from dataReader import DataReader
import numpy as np
from sklearn.manifold import MDS
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class HMMRunner:
    def __init__(self,data_dir,player_file,ncomp=2):
        self.reader = DataReader(data_dir,player_file)
        self.n = ncomp
        #can change this for convergence
        self.iters = 750
        self.resets = 5 #number of resets used to get good gradient approximations of HMM
        self.colors = np.array(['r','b'])

        self.hmmObj = [hmm.GaussianHMM(n_components=ncomp,n_iter=self.iters) for _ in range(self.resets)]
        #rEAding data
        self.X, self.y, self.lengths = self.reader.get_data()

        #fitting model to read data
        scores = []
        predictions = []
        for x in range(self.resets):
            self.hmmObj[x].fit(self.X[:2050,:],self.lengths[:25])
            predictions.append(self.hmmObj[x].predict(self.X,self.lengths))
            pos_train_acc = np.sum(predictions[x] == self.y)/float(len(self.y))
            neg_train_acc = np.sum((1-predictions[x]) == self.y)/float(len(self.y))
            scores.append(max(pos_train_acc,neg_train_acc))
        scores = np.array(scores)
        predictions = np.array(predictions)


        print self.X.shape
        idx = np.argmax(scores)
        self.hmmObj = self.hmmObj[idx]
        self.train_acc = scores[idx]
        self.predictions = predictions[idx]

        #fitting linear svm
        self.lin_svm = LinearSVC()
        self.log_reg = LogisticRegression()
        self.gmm = GaussianMixture(n_components=ncomp, n_init=5)
        
        train_ind = np.random.choice(self.X.shape[0],2000,replace=False)
        test_ind = np.delete(np.arange(self.X.shape[0]),train_ind)


        self.lin_svm.fit(self.X[train_ind,:],self.y[train_ind])
        self.lin_acc = self.lin_svm.score(self.X[test_ind,:],self.y[test_ind])

        self.log_reg.fit(self.X[train_ind,:], self.y[train_ind])
        self.log_acc = self.log_reg.score(self.X[test_ind,:],self.y[test_ind])

        self.gmm.fit(self.X[train_ind,:],self.y[train_ind])
        self.gmm_predictions = self.gmm.predict(self.X[test_ind,:])
        pos_gmm_acc =  np.sum(self.gmm_predictions == self.y[test_ind])/float(len(self.y[test_ind]))
        neg_gmm_acc = np.sum((1-self.gmm_predictions) == self.y[test_ind])/float(len(self.y[test_ind]))

        if pos_gmm_acc > neg_gmm_acc:
            self.polarity = 1
            self.gmm_acc = pos_gmm_acc
        else:
            self.polarity = 0
            self.gmm_acc = neg_gmm_acc

    def predict(self,data):
        return self.hmmObj.predict(data)

    def accuracy(self,data,label):
        pred = self.predict(data)
        pos_acc = np.sum(pred == label)/float(len(label))
        neg_acc = np.sum((1-pred)==label)/float(len(label))
        if self.polarity == 1:
            return pos_acc
        return neg_acc

    def get_true_transmat(self):
        a = np.zeros((2,2))
        print len(self.lengths)
        print self.lengths[0]
        for l in range(len(self.lengths)):
            y = self.y[82*l:82*(l+1)]
            for k in range(len(y)-1):
                a[y[k]][y[k+1]] += 1.
        a /= (82.*len(self.lengths))
        a[0] /= np.sum(a[0])
        a[1] /= np.sum(a[1])
        return a


    def tp_rate(self,data,label):
        pred = self.predict(data)
        if self.polarity == 0:
            pred = 1 - pred
        count = 0
        for x in range(len(label)):
            if label[x] == 1 and pred[x] == 1:
                count += 1
        return float(count)/len(label[label==1])

    def tn_rate(self,data,label):
        pred = self.predict(data)
        if self.polarity == 0:
            pred = 1 - pred
        count = 0
        for x in range(len(label)):
            if label[x] == 0 and pred[x] == 0:
                count += 1
        return float(count)/len(label[label==0])

    def get_team_accuracy(self):
        for t in range(len(self.reader.teams)):
            team = self.reader.teams[t]
            pred = self.predict(self.X[82*t:82*(t+1),:])
            if self.polarity==0:
                pred = 1 - pred
            acc= self.accuracy(self.X[82*t:82*(t+1),:],self.y[82*t:82*(t+1)])
            tn_acc = self.tn_rate(self.X[82*t:82*(t+1),:],self.y[82*t:82*(t+1)])
            tp_acc = self.tp_rate(self.X[82*t:82*(t+1),:],self.y[82*t:82*(t+1)])

            print "Team: " + str(team) + "  Acc: " + str(acc)
            print "Team: " + str(team) + "  Wins: " + str(np.sum(pred))
            print "Team: " + str(team) + "  TP: " + str(tp_acc)
            print "Team: " + str(team) + "  TN: " + str(tn_acc)
            print


    def log_ll(self,data):
        return self.hmmObj.score(data)
    
    def plot_gt(self):
        coords = MDS(n_components=2).fit_transform(self.X)
        plt.scatter(coords[:,0],coords[:,1],c=self.colors[self.y],marker='x')
        plt.show()
        return

    def plot_hmm_preds(self):
        coords = MDS(n_components=2).fit_transform(self.X)
        plt.scatter(coords[:,0],coords[:,1], c=self.colors[self.predictions],marker='x')
        plt.show()
        return

    def plot_gmm_preds(self):
        coords = MDS(n_components=2).fit_transform(self.X)
        plt.scatter(coords[:,0],coords[:,1], c=self.colors[self.gmm_predictions],marker='x')
        plt.show()
        return

if __name__=="__main__":
    runner = HMMRunner("../data/2016_adv/","../data/players/2016_adv.csv")
    print "Accuracy: " + str(runner.train_acc)
    print "Linear SVM Accuracy: " + str(runner.lin_acc)
    print "Logistic Regression Accuracy: " + str(runner.log_acc)
    print "GMM Accuracy: " + str(runner.gmm_acc)    
    
    runner.get_team_accuracy()
    print 
    print runner.polarity
    print runner.hmmObj.transmat_
    print 
    print runner.get_true_transmat()


    


    diff = runner.hmmObj.means_[0] - runner.hmmObj.means_[1]
    #X = runner.reader.prepro.transform(runner.X)
    #y = runner.log_reg.predict(runner.X)
    #mu0 = np.mean(X[y == 0],axis=0)
    #mu1 = np.mean(X[y == 1],axis=0)
    #diff = mu1 - mu0


    bins = np.arange(len(diff))
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)    
    ax.bar(bins,diff, align='center')
    ax.set_xticks(np.arange(len(diff)))
    ax.set_xticklabels(runner.reader.featureNames,rotation='vertical')
    ax.set_ylabel("Abs. Difference")
    plt.savefig("temp.png")
    #runner.plot_hmm_preds()
    #runner.plot_gmm_preds()
