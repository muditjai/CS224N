import cPickle as pickle

with open("saved_params_%d.npy" % 40000, "r") as f:
    params = pickle.load(f)
    state = pickle.load(f)

with open("saved_params_%d_binary.npy" % 40000, "wb") as f:
    pickle.dump(params, f)
    pickle.dump(state, f)