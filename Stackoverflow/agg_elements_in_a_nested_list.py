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




# A broader solution using loop:

def merge_result(data):
    ar = []
    for i in range(0, len(data)-2, 3):
        temp = []        
        for j in range(len(data[i])):
            if data[i][j] == data[i+1][j]:
                temp.append(data[i][j])
            elif data[i+2][j] == data[i+1][j]:
                temp.append(data[i+1][j])
            elif data[i][j] == data[i+2][j]:
                temp.append(data[i][j])
            else:
                temp.append('O')
        ar.append(temp)
    return ar

if __name__ == "__main__":
    original_list = [['B_S', 'O', 'O', 'O'],
                 ['B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'B_S', 'O'],

                 ['O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'O'],

                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'B_S', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O'],
                 ['O', 'O', 'O', 'O', 'O', 'B_S', 'O', 'O', 'B_S', 'O', 'O', 'O']]
    print(merge_result(original_list))
