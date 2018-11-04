import pickle

# pickle list object

numbers_list = [1, 2, 3, 4, 5]
list_pickle_path = 'list_pickle.pkl'

# Create an variable to pickle and open it in write mode
list_pickle = open(list_pickle_path, 'wb')
pickle.dump(numbers_list, list_pickle)
list_pickle.close()