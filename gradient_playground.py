import numpy as np

def brute_force_gradient(func, input, epsilon):
    gradient = np.zeros(len(input))
    for i in range(input.size):
        print(i)
        print(input)
        temp_array = np.copy(input)
        temp_array[i] = temp_array[i] + epsilon
        print(temp_array)
        numerator = func(temp_array) - func(input)
        print(numerator)
        denominator = epsilon
        print(denominator)
        gradient[i] = numerator/denominator
    return gradient

def better_gradient(f, input, epsilon):
    gradient = np.zeros_like(input)
    for i in range(input.size):
        single_epsilon = np.zeros_like(input)
        single_epsilon[i] = epsilon
        print(single_epsilon)
        gradient[i] = (f(input+single_epsilon)-f(input))/epsilon
    return gradient

def older_better_gradient(f, input, epsilon):
    gradient = np.zeros_like(input)
    iterator = np.nditer(input, flags=['multi_index'])
    for i in iterator:
        eps = np.zeros_like(input)
        eps[iterator.multi_index] = epsilon
        gradient[iterator.multi_index] = (f(input+eps)-f(input))/epsilon
    return gradient    

def even_better_gradient(f, input, epsilon):
    gradient = np.zeros_like(input)
    epsilon_comb = np.zeros_like(input)
    iter = np.nditer(epsilon_comb, flags=['multi_index'])
    while not iter.finished:
        print('\n ## new pass, index: ', iter.multi_index )
        epsilon_comb[iter.multi_index] = epsilon
        print(' #### eps\n',epsilon_comb)
        gradient[iter.multi_index] = (f(input+epsilon_comb)-f(input))/epsilon
        epsilon_comb[iter.multi_index] = 0
        print('## partial for this cell', gradient[iter.multi_index])
        print(' #### eps\n',epsilon_comb)
        print('\n') 
        is_not_finished = iter.iternext()
    return gradient    

def sample_function(array_form):
    x = np.reshape(array_form,-1)
    return 3*x[0] + 2*x[1]*x[1] - x[2]

def f2(x):
    return 1

x = np.array([1.5, 1, 3])

print(x)
r = sample_function(x)
print(r)
#g = brute_force_gradient(sample_function,x,0.001)
#print(g)
print('')
g = even_better_gradient(sample_function,x,0.001)
print(g)

print('\n\n')
x = np.arange(12).reshape(2,3,2) + 1
x = np.array([1.5, 1, 3, 5, 1, 3, 1.5, 1, 3, 5, 1, 3]).reshape(2,3,2) + 3
rng = np.random.default_rng()
x = rng.standard_normal((13002,1))
print(x)
print('')
g = even_better_gradient(sample_function,x,0.001)
print(g)
