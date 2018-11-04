import pickle
# unpickling the list object

# Need to open the pickled list object into read mode

list_pickle_path = 'list_pickle.pkl'
list_unpickle = open(list_pickle_path, 'rb')

# load the unpickle object into a variable
numbers_list = pickle.load(list_unpickle)

print(numbers_list)