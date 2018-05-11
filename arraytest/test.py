import os
import pickle

parvalue = os.environ['SGE_TASK_ID']

# output to log file
print("Parameter value: " + str(parvalue) + "\n")

# create a pickle file with customized filename
fname = "output" + str(parvalue) + ".pkl"
with open(fname,'wb') as f:
    pickle.dump([parvalue], f)



