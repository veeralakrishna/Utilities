'''
The hardest part of the problem is just grouping your lists into the three chunks:

>>> [original_list[i*3:i*3+3] for i in range(len(original_list)//3)]
[[['B_S', 'O', 'O', 'O'], ['B_S', 'O', 'O', 'O'], ['O', 'O', 'B_S', 'O']], 
[['O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O']], 
[['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'B_S', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O']]]
Once you have that, though, you can use zip to group the items you want to compare:

>>> chunks = [original_list[i*3:i*3+3] for i in range(len(original_list)//3)]
>>> [list(zip(*j)) for j in chunks]
[[('B_S', 'B_S', 'O'), ('O', 'O', 'O'), ('O', 'O', 'B_S'), ('O', 'O', 'O')],
[('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('B_S', 'O', 'O'), ('O', 'B_S', 'O'), ('O', 'O', 'B_S'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O')], 
[('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('B_S', 'B_S', 'B_S'), ('O', 'O', 'O'), ('O', 'O', 'O'), ('B_S', 'B_S', 'B_S'), ('B_S', 'O', 'O'), ('O', 'O', 'O'), ('O', 'O', 'O')]]
And then you just want to pick the item in each of those zipped tuples that appears the most frequently -- aka statistics.mode:

>>> import statistics
>>> [[statistics.mode(i) for i in zip(*j)] for j in chunks]
[['B_S', 'O', 'O', 'O'], 
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O']]
or all together:
'''
  
from statistics import mode

original_list = [['B_S', 'O', 'O', 'O'],
                 ['B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'B_S', 'O'],

                 ['O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O'],

                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'B_S', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O']]
CHUNK_LEN = 3

desired_output = [
    [mode(i) for i in zip(*j)] 
    for j in (
        original_list[i*CHUNK_LEN:(i+1)*CHUNK_LEN] 
        for i in range(len(original_list)//CHUNK_LEN)
    )
]

# Obviously if you can make original_list just be correctly grouped up front it's a lot easier:

from statistics import mode

original_list = [[['B_S', 'O', 'O', 'O'],
                 ['B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'B_S', 'O']],

                 [['O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O']],

                 [['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'B_S', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O']]]

desired_output = [
    [mode(i) for i in zip(*j)] 
    for j in original_list
]
