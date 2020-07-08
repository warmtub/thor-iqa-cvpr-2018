import sys
import numpy as np
a = np.load(sys.argv[1])
a = 4*a
a = a.astype('int')
print('   ', end='')
for i in range(np.min(a[:,1]), np.max(a[:,1]), 1):
    print('%2d'%i, end='')
print('')
for i in range(np.min(a[:,0]), np.max(a[:,0]), 1):
    print('%3d'%i, end='')
    for j in range(np.min(a[:,1]), np.max(a[:,1]), 1):
        if np.any(np.all(a==[i,j], axis=1)):
            print(' .', end='')
        else:
            print(' *', end='')
    print('')
