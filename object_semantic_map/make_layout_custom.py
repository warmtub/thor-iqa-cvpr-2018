import numpy as np
import os

"""""""""""""""""""""""""""
↓↓↓CHANGE THIS PART ONLY↓↓↓
"""""""""""""""""""""""""""
out_name = "Lab01"

custom = [['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','.','.','.','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','.','.','.','.','.','.','.','*','*','*','*'],
          ['*','*','*','*','*','.','.','.','.','.','.','.','*','*','*','*'],
          ['*','*','*','*','*','.','.','.','.','.','.','.','*','*','*','*'],
          ['*','*','*','*','*','.','.','.','.','.','.','.','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*'],
          ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*']]


"""""""""""""""""""""""""""
↑↑↑CHANGE THIS PART ONLY↑↑↑
"""""""""""""""""""""""""""

layout = np.zeros((12, 16))

for i in range(layout.shape[0]):
    for j in range(layout.shape[1]):
        if custom[i][j] == '.': layout[i][j] = 1

grids = np.empty((0,2), int)
for i in range(layout.shape[0]):
    for j in range(layout.shape[1]):
        if layout[i,j]:
            print('*', end ='')
            grids = np.append(grids, np.array([[j,i]]), axis=0)
        else:
            print('.', end ='')
    print("")

print (grids)
out_name = 'layouts/' + out_name + '-layout.npy'
with open(out_name, 'wb') as f:
    np.save(f, grids)
    print('Write success')
