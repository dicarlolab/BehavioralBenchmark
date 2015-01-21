M = np.random.randint(2,size = (128,40)) # Hypothetical Model 128 Images, 40 instantiations 
n_boots = 10
Consistencies,normalized_humans_to_pool_consistencies = empirical_consistency(M,n_boots)

hist(Consistencies)
hist(normalized_humans_to_pool_consistencies )
