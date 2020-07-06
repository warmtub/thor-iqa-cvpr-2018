import numpy as np
import os

"""""""""""""""""""""""""""
↓↓↓CHANGE THIS PART ONLY↓↓↓
"""""""""""""""""""""""""""

out_name = "Lab01"

ldcorners = [[5, 0, 1],
             [3, 6, 0],
             [11, 12, 1],
             [13, 12, 0]]


"""""""""""""""""""""""""""
↑↑↑CHANGE THIS PART ONLY↑↑↑
"""""""""""""""""""""""""""

items_size = [[4, 6],
              [4, 10],
              [4, 10],
              [4, 4]]

layout = np.zeros((12, 16))

for i in range(16):
    layout[0, i] = 1
    layout[11, i] = 1
for i in range(12):
    layout[i, 0] = 1
    layout[i, 15] = 1

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
