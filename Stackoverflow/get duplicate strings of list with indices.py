# This uses collections.defaultdict and enumerate to efficiently collect the indeces of each word. 
# Ridding this of duplicates remains a simple conditional comprehension or loop with an if statement.

sample = """An article is any member of a class of dedicated words that are used with noun phrases to
mark the identifiability of the referents of the noun phrases. The category of articles constitutes a
part of speech. In English, both "the" and "a" are articles, which combine with a noun to form a noun
phrase."""

sample_list = sample.split()

my_list = [x.lower() for x in sample_list]



from collections import defaultdict

def get_duplicates(my_list):
    indeces = defaultdict(list)

    for i, w in enumerate(my_list):
        indeces[w].append(i)

    for k, v in indeces.items():
        if len(v) > 1:
            print(k, v)
            
            
# https://stackoverflow.com/questions/66136490/how-to-get-duplicate-strings-of-list-with-indices-in-python
