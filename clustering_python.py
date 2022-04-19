# -*- coding: utf-8 -*-

#modification du dossier par défaut
import os
os.chdir("D:/_Travaux/university/Cours_Universite/Supports_de_cours/Informatique/Python/Tutoriels")

#importation des données
import pandas
fromage = pandas.read_table("fromage.txt",sep="\t",header=0,index_col=0)

#dimension des données
print(fromage.shape)

#6 premières lignes des données
print(fromage.iloc[0:6,:])

#statistiques descriptives
print(fromage.describe())

#graphique avec croisement deux à deux
from pandas.tools.plotting import scatter_matrix
scatter_matrix(fromage,figsize=(9,9))

#centrage réduction des données
from sklearn import preprocessing
fromage_cr = preprocessing.scale(fromage)

#librairies pour la CAH
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#générer la matrice des liens
Z = linkage(fromage_cr,method='ward',metric='euclidean')

#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=0)
plt.show()

#matérialisation des 4 classes (hauteur t = 7)
plt.title('CAH avec matérialisation des 4 classes')
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=7)
plt.show()

#découpage à la hauteur t = 7 ==> 4 identifiants de groupes obtenus
groupes_cah = fcluster(Z,t=7,criterion='distance')
print(groupes_cah)

#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)

#affichage des observations et leurs groupes
print(pandas.DataFrame(fromage.index[idg],groupes_cah[idg]))

#k-means sur les données centrées et réduites
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(fromage_cr)

print(kmeans.inertia_)

#index triés des groupes
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pandas.DataFrame(fromage.index[idk],kmeans.labels_[idk]))

#distances aux centres de classes des observations
print(kmeans.transform(fromage_cr))

#correspondance avec les groupes de la CAH
pandas.crosstab(groupes_cah,kmeans.labels_)

#librairie pour évaluation des partitions
from sklearn import metrics

#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 10
res = np.arange(9,dtype="double")
for k in np.arange(9):
    km = cluster.KMeans(n_clusters=k+2)
    km.fit(fromage_cr)
    res[k] = metrics.silhouette_score(fromage_cr,km.labels_)
    
print(res)

#graphique
import matplotlib.pyplot as plt
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()

#moyenne par variable
m = fromage.mean()

#TSS
TSS = fromage.shape[0]*fromage.var(ddof=0)
print(TSS)

#data.frame conditionnellement aux groupes
gb = fromage.groupby(kmeans.labels_)

#effectifs conditionnels
nk = gb.size()
print(nk)

#moyennes conditionnelles
mk = gb.mean()
print(mk)

#pour chaque groupe ecart à la moyenne par variable
EMk = (mk-m)**2

#pondéré par les effectifs du groupe
EM = EMk.multiply(nk,axis=0)

#somme des valeurs => BSS
BSS = np.sum(EM,axis=0)
print(BSS)

#carré du rapport de corrélation
#variance expliquée par l'appartenance aux groupes
#pour chaque variable
R2 = BSS/TSS
print(R2)

#ACP
from sklearn.decomposition import PCA
acp = PCA(n_components=2).fit_transform(fromage_cr)

#projeter dans le plan factoriel
#avec un code couleur selon le groupe
for couleur,k in zip(['red','blue','lawngreen','aqua'],[0,1,2,3]):
    plt.scatter(acp[kmeans.labels_==k,0],acp[kmeans.labels_==k,1],c=couleur)
plt.show()    

#retirer des observations le groupe n°0 des k-means
fromage_subset = fromage.iloc[kmeans.labels_!=0,:]
print(fromage_subset.shape)
    
#centrer et réduire
fromage_subset_cr = preprocessing.scale(fromage_subset)

#générer la matrice des liens
Z_subset = linkage(fromage_subset_cr,method='ward',metric='euclidean')

#cah et affichage du dendrogramme
plt.title("CAH")
dendrogram(Z_subset,labels=fromage_subset.index,orientation='left',color_threshold=7)
plt.show()

#groupes
groupes_subset_cah = fcluster(Z_subset,t=7,criterion='distance')
print(groupes_subset_cah)

#ACP
acp_subset = PCA(n_components=2).fit_transform(fromage_subset_cr)

#projeter dans le plan factoriel
#avec un code couleur selon le groupe
plt.figure(figsize=(18,7.715))
for couleur,k in zip(['blue','lawngreen','aqua'],[1,2,3]):
    plt.scatter(acp_subset[groupes_subset_cah==k,0],acp_subset[groupes_subset_cah==k,1],c=couleur)

#mettre les labels des points
for i,label in enumerate(fromage_subset.index):
    plt.annotate(label,(acp_subset[i,0],acp_subset[i,1]))

plt.show()      