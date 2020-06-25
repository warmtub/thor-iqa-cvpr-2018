import numpy as np
import os

"""""""""""""""""""""""""""
↓↓↓CHANGE THIS PART ONLY↓↓↓
"""""""""""""""""""""""""""

out_name = "Lab01"

ldcorners = [[3, 0, 1],
             [1, 8, 0],
             [9, 14, 1],
             [11, 14, 0]]


"""""""""""""""""""""""""""
↑↑↑CHANGE THIS PART ONLY↑↑↑
"""""""""""""""""""""""""""

items_size = [[2, 4],
              [2, 8],
              [2, 8],
              [2, 2]]

layout = np.zeros((12, 16))

for i in range(len(ldcorners)):
    if ldcorners[i][2]:
        layout[ldcorners[i][0]-items_size[i][1]+1:ldcorners[i][0]+1,
               ldcorners[i][1]:ldcorners[i][1]+items_size[i][0]] = 1
    else:
        layout[ldcorners[i][0]-items_size[i][0]+1:ldcorners[i][0]+1,
               ldcorners[i][1]:ldcorners[i][1]+items_size[i][1]] = 1

grids = np.empty((0,2), int)
for i in range(layout.shape[0]):
    for j in range(layout.shape[1]):
        if layout[i,j]:
            print('*', end ='')
        else:
            print('.', end ='')
            grids = np.append(grids, np.array([[i,j]]), axis=0)
    print("")

print (grids)
out_name = 'layouts/' + out_name + '-layout.npy'
with open(out_name, 'wb') as f:
    np.save(f, grids)
