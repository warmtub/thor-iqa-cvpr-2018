import os
import argparse
import itertools
import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib import colorbar
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import constants

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="npy file path")

args = parser.parse_args()

#np_map = np.load(args.f)
#print(np_map)
#print(os.path.join(args.f, "*.npy"))

for file in glob.glob(os.path.join(args.f, "*.npy")):
    print(file)
    #if "FloorPlan1" not in file: continue
    np_map = np.load(file)
    #fig, ax = plt.subplots(1, 1)
    #obj_id = 19
    #pcm = ax.pcolor(np_map[:,:,obj_id], cmap=plt.get_cmap('Reds'), vmin=-0, vmax=1)
    #print(np_map[np_map[:,:,obj_id]>0])
    #ax.set_title(constants.OBJECTS[obj_id], fontsize=5)
    #ax.axis('off')
    fig, axs = plt.subplots(4, 5)
    for (plot_i, plot_j) in list(itertools.product("0123","01234")):
        #print (i, j)
        obj_id = (int(plot_i))*5+int(plot_j)+1
        if obj_id > 19: obj_id = 0
        #if obj_id == 0: continue
        ax = axs[int(plot_i), int(plot_j)]
        pcm = ax.pcolor(np_map[:,:,obj_id+1], cmap=plt.get_cmap('Reds'), vmin=-0, vmax=1)
        ax.set_title(constants.OBJECTS[obj_id], fontsize=5)
        ax.axis('off')
        
        """
        sil = []
        
        mask = np.array(np.where(np_map[:,:,obj_id] > 0)).transpose()
        
        if mask.shape[0] == 0:
            continue
        elif mask.shape[0] == 1:
            centers = [mask[0]]
            pcm = ax.scatter(centers[:,1], centers[:,0], s = 1.0, alpha = 0.5)
        kmax = min(10 , mask.shape[0]-1)

        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(1, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(mask)
            labels = kmeans.labels_
            #print(k)
            #print(np_map[:,:,obj_id].shape)
            #print(labels)
            if np.unique(labels).shape[0] == 1:
                sil.append(0)
                continue
            sil.append(silhouette_score(mask, labels, metric = 'euclidean')) 
            #print("centers: ", kmeans.cluster_centers_)
        #print(constants.OBJECTS[obj_id])
        if len(sil) == 0:
            continue
        k = sil.index(max(sil))+2
        print(k)
        kmeans = KMeans(n_clusters = k).fit(mask)
        centers = kmeans.cluster_centers_
        print(centers)
        pcm = ax.scatter(centers[:,1], centers[:,0], s = 1.0, alpha = 0.5)
        
	"""
        
    #fig.tight_layout()
    #fig.colorbar(pcm, ax=axs[:])
    #fig_path = os.path.join(prefix, str(index)+'.png')
    #print(args.f.split('.')[0])
    mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())

    #plt.show()
    plt.tight_layout()
    plt.savefig("%s.png" % (file.split('.')[0]), dpi = 1000)
    print("%s.png" % (file.split('.')[0]), "saved")
    plt.close()
    #break
