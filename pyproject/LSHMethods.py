# What kind of methods are necessary for LSH
# Also, this must extend anndata objects so it
# can interface with scanpy
import numpy as np
import scanpy as sc


# param table-the hashtable
# param vector- the vector being inserted
# param random_vector-the random_vector that we are projecting onto for that round of hashing
# param bucket_width-the sensitivity
# return it is void but it inserts the vector into the hashtable
def insertToTable(table, vector, random_vector, bucket_width):
    hash_index = np.floor(np.dot(vector, random_vector) / bucket_width)
    table[hash_index].append(vector)
    return hash_index % len(table)


# This method hashes the new dataset and the library
# param file-the name of the anndata object holding the new dataset
# param buckets-the size of the hashtable
# param library_file name of the anndata object holding the library file
# returns a hashtable(w/ chaining) and which buckets contain the library
# this method is not used now that I made a hashing object
def hash_entire_dataset(file, buckets, library_file):
    ad = sc.read(file)
    library = sc.read(library_file)
    random_Hash_Vector = np.random.rand(1, ad.n_obs)
    hashtable = [[] for _ in range(buckets)]
    library_indices = []

    for i in range(ad.n_vars):
        vec = ad.var_vector(i)
        insertToTable(hashtable, vec, random_Hash_Vector)
    for i in range(library.n_vars):
        vec_lib = library.var_vector(i)
        index = insertToTable(hashtable, vec_lib, random_Hash_Vector)
        library_indices.append(index)

    return hashtable, library_indices
