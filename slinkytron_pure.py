
class Slinkytron:
    """Abritrary depth perceptron network with non-linear activation"""

    def __init__(self,neuron_count):
        self.compute_layers = range(1,len(neuron_count))
        self.output_layer = len(neuron_count)-1
        self.W = []  # Weights 
        self.b = []  # biases
        self.z = []  # weighted inputs 
        self.a = []  # activations
        # the first layer is the input with no inbound weights, biases or sums
        self.W.append(None)
        self.b.append(None)
        self.z.append(None)
        input_layer_activations = []
        for j in range(0,neuron_count[0]):
            input_layer_activations.append(j)
        self.a.append(input_layer_activations)
        
        for L in self.compute_layers:
            new_layer_weights_matrix = []
            for j in range(0,neuron_count[L]):
                new_row = []
                for k in range(0, neuron_count[L-1]):
                    new_row.append(j+0.001*k)
                new_layer_weights_matrix.append(new_row)
            self.W.append(new_layer_weights_matrix)
            
            new_layer_biases = []
            for j in range(0,neuron_count[L]):
                new_layer_biases.append(j)
            self.b.append(new_layer_biases)
            
            new_layer_weighted_inputs = []
            for j in range(0,neuron_count[L]):
                new_layer_weighted_inputs.append(j)
            self.z.append(new_layer_biases)
            
            new_layer_activations = []
            for j in range(0,neuron_count[L]):
                new_layer_activations.append(j)
            self.a.append(new_layer_activations)

    def pretty_print(self):
        width = 10
        precision = 4
        print('')
        print('neuron layer 0 (input layer):')
        L = 0
        for j,row in enumerate(self.a[L]):
            label = f'a_{j}({L})'
            print(f'{label:7} = {float(self.a[L][j]):3.3}')
        for L in self.compute_layers:
            print('')
            print('neuron layer '+str(L)+':')

            for j,row in enumerate(self.W[L]):
                for k,col in enumerate(row):
                    label = f'w_{j},{k}({L})'
                    print(f'{label} = {round(self.W[L][j][k],3):{width}.{precision}}',end='  ')
                print('')

            for j,row in enumerate(self.a[L]):
                print('a_'+str(j)+'('+str(L)+') = '+str(self.a[L][j]))

# p = Slinkytron([9,2,2,1])
# print(p.W)
# print(p.b)
# print(p.z)
# print(p.a)