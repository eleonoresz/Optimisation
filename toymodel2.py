#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:52:05 2019

@author: eleonore
"""
import scipy.io
from random import normalvariate
import numpy as np;
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.spatial
import random
from random import uniform
from scipy.fftpack import fft
import scipy.stats
from collections import Counter
from matplotlib import animation, rc
from IPython.display import HTML, Image
from sklearn import model_selection
from threading import Thread
import pickle
import sys
 

listeDistance = []
for i in range(1,9):
    #a1 = np.loadtxt('/home/schiltz/Documents/Modeles Comp/Evaluation Modele/souris1_DistributionDistanceHabituation.csv',delimiter=',')
    listeDistance.append(np.loadtxt('souris{:d}_DistributionDistanceHabituation.csv'.format(i),delimiter=','))

# MODE

transition_cote = 1



N = 0
S = 1
E = 2
W = 3
NE = 4
NW = 5
SE = 6
SW = 7
NoOp = 8



l = 9
L = 9
n = l*L


temps = 1500

env = np.array([
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,0.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,0.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,0.0, 0.0, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

mapN = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapN[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i-1][j]):
            mapN[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapN[(i-1)*l+j-1][(i-2)*l+j-1] = 1.0
            
mapS = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapS[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i+1][j]):
            mapS[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapS[(i-1)*l+j-1][(i)*l+j-1] = 1.0
            
            
mapE = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i][j+1]):
            mapE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapE[(i-1)*l+j-1][(i-1)*l+j] = 1.0      
            
mapW = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i][j-1]):
            mapW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapW[(i-1)*l+j-1][(i-1)*l+j-2] = 1.0     
            
mapNE = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapNE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i-1][j+1]):
            mapNE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapNE[(i-1)*l+j-1][(i-2)*l+j] = 1.0      
            
mapNW = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapNW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i-1][j-1]):
            mapNW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapNW[(i-1)*l+j-1][(i-2)*l+j-2] = 1.0    
            
mapSE = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapSE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i+1][j+1]):
            mapSE[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapSE[(i-1)*l+j-1][(i)*l+j] = 1.0      
            
mapSW = np.zeros((n,n))
for i in range(1,10):
    for j in range(1,10):
        if np.isnan(env[i][j]):
            mapSW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        elif np.isnan(env[i+1][j-1]):
            mapSW[(i-1)*l+j-1][(i-1)*l+j-1] = 1.0
        else:
            mapSW[(i-1)*l+j-1][(i)*l+j-2] = 1.0    
            

            
# class of P & transition
            
class mdp():

    def __init__(self):
        self.nX = 81
        if transition_cote == 0:
            self.nU = 5
        else: 
            self.nU = 9
        self.P0 = np.zeros(self.nX,)
        self.P0[0] = 1
        self.P = np.empty((self.nX,self.nU,self.nX))
        self.level = -10
        self.P[:,N,:]=  mapN
        self.P[:,S,:]=  mapS
        self.P[:,W,:]=  mapW
        self.P[:,E,:]=  mapE
        if transition_cote !=0:
            self.P[:,NE,:]=  mapNE
            self.P[:,NW,:]=  mapNW
            self.P[:,SE,:]=  mapSE
            self.P[:,SW,:]=  mapSW
        
        self.P[:,NoOp,:]=  np.eye(self.nX)
             
        self.r = np.array([
                corner, walls, walls, walls, walls, walls, walls, walls, corner,
            	walls, other, other, other, other, other, other, other, walls,
            	corner, walls, walls, walls, walls, walls, walls,other, walls,
            	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, walls, walls,
            	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, walls, walls, 
            	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, walls, walls,
            	corner, walls, walls, walls, walls, walls, walls, other, walls,
            	walls,other, other, other, other, other, other,other, walls,
            	corner, walls, walls,walls, walls, walls, walls, walls, corner])

        self.gamma = 0.95;
        
    def MDPStep(self,x,u,i):
        y = self.discreteProb(self.P[x,u,:].reshape(self.nX,1))
        r = self.r[y] 
        return [y,r]
  

    def discreteProb(self,p):
        r = np.random.random()
        cumprob=np.hstack((np.zeros(1),p.cumsum()))
        sample = -1
        for j in range(p.size):
            if (r>cumprob[j]) & (r<=cumprob[j+1]):
                sample = j
                break
        return sample
    
    
    def QLearning(self,tau,position,seuil,tol):
        Q = np.zeros((self.nX,self.nU))
        nbIter = temps
        alpha = 0.2
        x = int(1)
        level = self.level
        rTotal = []
        for i in range(nbIter):
             
             u = self.discreteProb(self.softmax(Q,x,tau,level,seuil))
             [y,r] = self.MDPStep(x,u,i)
             Q[x,u] = Q[x,u] + alpha * (r + self.gamma * Q[y,:].max() - Q[x,u])
             position.append(x)
             x = y
             level+=r
             rTotal.append(r)
             if np.sum(rTotal[i-20:i])<tol and level>20:
                 level = 0
                 print('cycle')
             
        Qmax = Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)
        return [Qmax,pol,position]
        
    def aleatoire(self,position):
        nbIter = temps
        x = 1
        x = int(x)
        for i in range(nbIter):
             u = random.choice([0,1,2,3,4,5,6,7]) 
             [y,r] = self.MDPStep(x,u,i)
             position.append(x)
             x = y
        return position
    
    def probaSelection(self,lastD):
        if lastD!=NoOp:
            alist = [S, SE, E, NE, N, NW, W, SW]
        # réorganisation de la liste en fonction du choix d'avant
            if alist.index(lastD)<4:
                dist_moy = - 4 - alist.index(lastD)
            else:
                dist_moy = +4 - alist.index(lastD)
            newlist = [0,0,0,0,0,0,0,0]
            for i in range(0,len(alist)):
                newlist[dist_moy+alist.index(alist[i])] = alist[i]
            newlist.append(newlist[0])  
        else:
            newlist = []
        return newlist
        
        
    def vitesse(self,listeD,u):
        saut = False
        if listeD!=[]:
            if u == listeD[4]:
                threshS = 0.3
            elif u == listeD[3]:
                threshS = 0.2
            elif u == listeD[3]:
                threshS = 0.2
            else:
                threshS = 0
            if random.random()<threshS:
                saut = True
        return saut,u
        
        
    def PrefCoinMur(self,position):
        nbIter = temps
        x = int(1)
        rTotal = []
        lastU = N
        for i in range(nbIter):
             newlist = self.probaSelection(lastU)
             #u = self.selection(x, lastU, transition_cote)
             listeChoix = self.selectionbiais(x,newlist)
             u =   random.choice(listeChoix)
             [y,r] = self.MDPStep(x,u,i)
             [saut,u]  = self.vitesse(newlist,u)
             if saut:
                 x = y
                 [y,r] = self.MDPStep(x,u,i)
             position.append(x)
             x = y
             rTotal.append(r)
             lastU = u
        return position
    
    
    def selectionbiais(self,x,newlist):
        # biais de direction
        if newlist!=[]:
            D = Counter(newlist[int(random.vonmisesvariate(mu, kappa)*4/np.pi)] for _ in range(nG))
            D2 = sorted(D.elements())
        else:
            D2 = []
        # biais de position
        r = []
        for u in range(self.nU):
            y = self.discreteProb(self.P[x,u,:].reshape(self.nX,1))
            r.append(self.r[y]) 
        P = Counter({N: int(r[0]), S: int(r[1]), E: int(r[2]), W: int(r[3]), NE: int(r[4]), NW: int(r[5]), SE: int(r[6]), SW: int(r[7]), NoOp: int(r[8])})
        P2 = sorted(P.elements())
        
        # choix
        C = P2+D2
        return C
        
        

    def softmax(self,Q,x,tau,level,seuil):
        if level<seuil:
            tauQ = np.exp(Q[x,:])/tau
            p = tauQ / tauQ.sum()
        else:
            tauQ = np.exp(Q[x,:])/tau
            p = 1 - tauQ / tauQ.sum()
        return p


    def OccupationTrials(self,position):
        Map = np.zeros((81,1))
        for i in range(temps):
            Map[position[i]] = Map[position[i]]+1
        Map2 = np.reshape(Map,(9,9))
        return Map2
    
    def DistanceTrials(self,Map2):
        coin = [[0,0],[8,8],[0,8],[8,0],[2,0],[6,0]]
        distanceTotal = []
        for i in range(9):
            for j in range(9):
                distance = []
                for c in range(6):
                    distance.append(scipy.spatial.distance.pdist([coin[c],[i,j]]))
                if Map2[i,j]!=0:
                    for ind in range(int(Map2[i,j])):
                        distanceTotal.append(np.min(distance))
        return distanceTotal
        
    def DistanceTrials2(self,position):
        coin = [[0,0],[8,8],[0,8],[8,0],[2,0],[6,0]]
        distanceTotal = []
        for i in range(len(position)):
            Map = np.zeros((81,1))
            Map[position[i]] = Map[position[i]]+1
            Map2 = np.reshape(Map,(9,9))
            xy = np.where(Map2 ==1)
            distance = []
            for c in range(len(coin)):
                distance.append(scipy.spatial.distance.pdist([coin[c],[xy[0][0],xy[1][0]]]))
            distanceTotal.append(np.min(distance))
        return distanceTotal
                    
    
    def moyenneNTrials(self,position, trials):
        occupationT = np.zeros((9,9))
        positionT   = []
        for i in range(trials):
            position = self.PrefCoinMur(position)
            positionT = positionT+position
            occupation = self.OccupationTrials(position)
            position = []
            occupationT = occupationT+occupation
        distanceT = self.DistanceTrials2(positionT)
        return occupationT,distanceT
            
    def evaluation(self,distanceM,distance2):
        S = []
        for i in range(len(distance2)):
            a1 = np.histogram(distanceM,normed = True)
            a2 = np.histogram(distance2[i],normed = True)
            S.append(scipy.stats.entropy(a1[0],a2[0])) 
        return S
        
        
#corner = 9
#walls  = 5
#other  = 2


## grid search pour les paramètres
#param_grid = {'walls': np.linspace(5,10,num=5),'corner': np.linspace(5,10,num=5),'other': np.linspace(5,10,num=5),'nG': np.linspace(0,1000,num=1001),'mu': np.linspace(0,2*np.pi,num=100),'kappa': np.linspace(0,100,num=150)}
#liste = model_selection.ParameterGrid(param_grid)
#
### #sélection du meilleur set de paramètre
#listeDivergence = np.zeros((len(liste),1))
liste = sys.argv[1:13]
trials = 100

#for i in range(len(liste)):

walls = float(liste[7][0:-1])
other = float(liste[9][0:-1])
corner = float(liste[11][0:-1])
mu    = float(liste[5][0:-1])
kappa = float(liste[1][0:-1])
nG = int(float(liste[3][0:-1]))

m = mdp()
position=[]
occupation, distanceM = m.moyenneNTrials(position,trials)
divergence = m.evaluation(distanceM,listeDistance)

indice = (walls,other,corner,mu,kappa,nG)
title = ('sortie{}.txt').format(indice)
with open(title, 'w') as file:
    file.write('divergence\n')
    file.write(str(divergence) + '\n')
    file.write('distance au coin \n')
    file.write(str(distanceM) + '\n')
    file.write('occupation\n')
    file.write(str(occupation) + '\n')
file.close()

#listeDivergence[i][0] = np.mean(divergence)
# np.save('listeDivergence')
#mindiff = np.min(listeDivergence[np.nonzero(listeDivergence)])
#print(liste[np.where(listeDivergence==mindiff)[0]])
#indice = np.where(listeDivergence[np.nonzero(listeDivergence)])#<0.03)

### résultats des minimums locaux
#for mindiff in indice[0]:
##    minlocal = np.where(listeDivergence==mindiff)[0]
#    corner = liste[mindiff].get('corner')
#    walls = liste[mindiff].get('walls')
#    other = liste[mindiff].get('other')
#    mu    = int(liste[mindiff].get('mu'))
#    kappa = int(liste[mindiff].get('kappa'))
#    nG = int(liste[mindiff].get('nG'))
#    m = mdp()
#    position=[]
#    trials = 10
#    occupation, distanceM = m.moyenneNTrials(position,trials)
#    divergence = m.evaluation(distanceM,listeDistance)
#    
#    #with open('listeDivergence', 'wb') as f:
#    #    pickle.dump(listeDivergence, f)
#        
#    #with open('listeDivergence', 'rb') as f:
#    #    listeDivergence = pickle.load(f)
#plt.figure()
#plt.plot(distanceM)
#plt.title('Distance au coin au cours du temps (optimisation 2)')
#plt.ylabel('Distance au coin')
#plt.xlabel('temps (par 500ms')
#plt.figure()
#plt.hist(distanceM)
#plt.title('Distribution des distances au coin (optimisation 2)')
#plt.ylabel('Occurence (500ms)')
#plt.xlabel('Distance au coin')
#plt.figure()
#plt.imshow(occupation,interpolation='None')
#plt.title('Occupation du labyrinthe (optimisation 2)')


#############################################"""""""""""#########"
#start = 0
#end = 100
#class Optimisation(Thread):
#
#    """Thread chargé simplement d'afficher une lettre dans la console."""
#
#    def __init__(self,start,end):
#        Thread.__init__(self)
#        cornerl = []
#        wallsl = []
#        otherl = []
#        for i in range(start,end):
#            cornerl.append(liste[i].get('corner'))
#            wallsl.append(liste[i].get('walls'))
#            otherl.append(liste[i].get('other'))
#
#
#    def run(self):
#        trials = 1
#        listeDivergence = np.zeros((len(otherl),1))
#        for i in range(len(otherl)):
#            corner = cornerl[i]
#            walls  = wallsl[i]
#            other  = otherl[i]
#            m = mdp()
#            position=[]
#            occupation, distanceM = m.moyenneNTrials(position,trials)
#            divergence = m.evaluation(distanceM,listeDistance)
#            listeDivergence[i][0] = np.mean(divergence)
#        mindiff = np.min(listeDivergence[np.nonzero(listeDivergence)])
#        print(liste[np.where(listeDivergence==mindiff)])
#
## Création des threads
#
#thread_1 = Optimisation(0,5)
#
#thread_2 = Optimisation(6,10)
#
#
## Lancement des threads
#
#thread_1.start()
#
#thread_2.start()
#
#
## Attend que les threads se terminent
#
#thread_1.join()
#
#thread_2.join()

##########################################################"


###[V,pol,position] = m.QLearning(0.5,position,seuil,tol)
###position = m.aleatoire(position)
###position = m.PrefCoinMur(position)

#plt.hist(distanceM,normed = True)
#plt.show()
#a1 = np.histogram(distanceM,normed = True)
##plt.imshow(occupation)
##plt.colorbar()
##plt.show()
##
##plt.hist(distance)
##plt.show()
#positionU = []
#positionU = m.aleatoire(positionU)
#occupationU = m.OccupationTrials(positionU)
#distanceU = m.DistanceTrials(occupationU)
#a2 = np.histogram(distanceU,normed = True)
#
#plt.hist(distanceU)
#plt.show()
#
#Map = np.zeros((9,9))
#Map2 = Map
#Map2[Map == 0.0] = np.nan 
#coin = [[0,0],[8,8],[0,8],[8,0],[2,0],[6,0]]
#distanceTotal = []
#for v in range(25):
#    for i in range(3):
#        for j in range(9):
#            distanceu = []
#            for c in range(6):
#                distanceu.append(scipy.spatial.distance.pdist([coin[c],[i,j]]))
#            Map2[i,j] = np.min(distanceu)
#            distanceTotal.append(np.min(distanceu))
#    for i in range(6,9):
#        for j in range(9):
#            distanceu = []
#            for c in range(6):
#                distanceu.append(scipy.spatial.distance.pdist([coin[c],[i,j]]))
#            Map2[i,j] = np.min(distanceu)
#            distanceTotal.append(np.min(distanceu))
#    for i in range(3,6):
#        for j in range(7,9):
#            distanceu = []
#            for c in range(6):
#                distanceu.append(scipy.spatial.distance.pdist([coin[c],[i,j]]))
#            Map2[i,j] = np.min(distanceu)
#            distanceTotal.append(np.min(distanceu))
#plt.hist(distanceTotal)
#plt.show()
#a3 = np.histogram(distanceTotal,normed = True)
#


# Distance de Kullbach Leibler


#plt.imshow(Map2)
#plt.colorbar()
#plt.title('Distance au coin en fonction de la position')
#plt.show()
                
#print(V)
#print(pol) 
#A = np.asarray(V)
#A = np.reshape(A,(9,9))
#plt.imshow(A)
#plt.title('Attirance naturelle coin - mur dans le U-maze')
#plt.show()


########################################################################################################################



#Map = np.zeros((81,1))
#for i in range(temps):
#    Map[position[i]] = Map[position[i]]+1
#Map2 = np.reshape(Map,(9,9))
#plt.imshow(Map2,interpolation='None')
#plt.colorbar()
#plt.title('Occupation du labyrinthe')
#plt.savefig('Biaisvitesse_Occupation')
#plt.show()
#
### distance coin -position
#
#
#coin = [[0,0],[8,8],[0,8],[8,0],[2,0],[5,0]]
#distanceTotal = []
#for i in range(9):
#    for j in range(9):
#        distance = []
#        for c in range(6):
#            distance.append(scipy.spatial.distance.pdist([coin[c],[i,j]]))
#        if Map2[i,j]!=0:
#            for ind in range(int(Map2[i,j])):
#                distanceTotal.append(np.min(distance))
#
#plt.hist(distanceTotal)
#plt.title('Distance au coin')
#plt.savefig('Biaisvitesse_DistanceCoin')
#plt.show()       
#
### cyclicité
#
#distanceTotal = []
#for i in range(len(position)):
#    x = np.ceil(position[i]/9);
#    y = np.mod(position[i],9);   
#    distance = []
#    for c in range(6):
#        distance.append(scipy.spatial.distance.pdist([coin[c],[x,y]]))
#    distanceTotal.append(np.min(distance))  
#plt.plot(distanceTotal)
#plt.title('Dynamique exploratoire')
#plt.savefig('Biaisvitesse_DynamiqueExploratoire.pdf')
#plt.show()

## transformée de fourier

#FFT = fft(distanceTotal)
#X = 2.0/temps * np.abs(FFT[0:temps/2])
#
#plt.plot(X)
#plt.title('FFT')
##plt.savefig('figureAleatoire4.pdf')
#plt.show()

# First set up the figure, the axis, and the plot element we want to animate

#Map = np.zeros((81,1))
#totalM = np.zeros((temps*9,9))
#ind = 0
#for i in range(temps):
#    Map = np.zeros((81,1))
#    Map[position[i]] = Map[position[i]]+1
#    alter = np.reshape(Map,(9,9))
#    for j in range(9):
#        for ij in range(9):
#            totalM[j+ind][ij] = alter[j][ij]
#    ind = ind+9
#
#
#Map2 = Map
#Map2[Map == 0.0] = np.nan 
#plt.imshow(Map2, interpolation='none')    
#plt.colorbar()    
#plt.title('U-maze')
#plt.savefig('Biaisvitesse_UMaze.pdf')
#plt.show()
#
#
#np.savetxt('maximums.txt', totalM)
