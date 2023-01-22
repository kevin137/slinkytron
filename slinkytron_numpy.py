import numpy as np

class Slinkytron:
    """Abritrary depth (non-linearly activated) perceptron network"""

    def __init__(self,neuron_count):
        self.compute_layers = range(1,len(neuron_count))
        self.output_layer = len(neuron_count)-1
        self.W = []  # Weights 
        self.b = []  # biases
        self.z = []  # weighted inputs 
        self.a = []  # activations
        # the first layer is the input with no inbound weights, biases or sums
        self.W.append([[np.nan]])
        self.b.append([[np.nan]])
        self.z.append([[np.nan]])
        self.a.append(np.zeros((neuron_count[0],1)))
        for i in self.compute_layers:
            self.W.append(np.random.random_sample(( neuron_count[i],
                                                    neuron_count[i-1]  )))
            self.b.append(np.random.random_sample(( neuron_count[i], 1 )))
            self.z.append(np.random.random_sample(( neuron_count[i], 1 )))
            self.a.append(np.random.random_sample(( neuron_count[i], 1 )))

    def one_shot(self,input_activation):
        if (input_activation.shape) != ((self.a[0]).shape):
            print('### input data has the wrong size, ', end='')
            print((input_activation.shape), end='')
            print(' instead of ', end='')
            print( ((self.a[0]).shape) )
            return
        #print("this looks like it will work")
        self.a[0] = input_activation
        for L in self.compute_layers:
            print("## processing",L,"th layer")
            self.z[L] = self.W[L] @ self.a[L-1] + self.b[L]
            self.a[L] = self.activation( self.z[L] )
            self.dump_network()
        return self.a[self.output_layer]

    def slantify(self):
        for i in self.compute_layers:
            print("## slantifying",i,"th layer")
            self.W[i][:] = i+1
            self.b[i][:] = i
    
    def activation(self,z):
        return 1/(1+np.exp(-z))  # sigmoid(z)
    
    def dee_activation_dee_z(self,z):
        return np.exp(z)/np.square(1+np.exp(z))  # ∂(sigmoid(z))/∂z

    def ideal(self,x):
        return np.atleast_2d(np.eye((len(self.a[-1])))[:,x]).T

    def single_symbol_cost(self,single_symbol_character_tuple):
        s,c = single_symbol_character_tuple
        return np.sum(np.square(self.one_shot(s) - self.ideal(c)))

    def cost_function(self,symbol_character_tuple_list):
        cost = np.zeros(len(symbol_character_tuple_list))
        for index,t in enumerate(symbol_character_tuple_list):
            cost[index] = self.single_symbol_cost(t)
        print('cost_function array')
        print(cost)
        return np.average(cost)

###########################################################################################3

    def dee_single_symbol_cost_dee_a(self,single_symbol_character_tuple):
        s,c = single_symbol_character_tuple
        self.one_shot(s)
        return np.sum(2*(np.square(self.a[self.output_layer] - self.ideal(c))))

    def dee_single_symbol_cost_dee_z(self,single_symbol_character_tuple):
        s,c = single_symbol_character_tuple
        self.one_shot(s)
        deeC0deea = np.sum(2*(np.square(self.a[self.output_layer] - self.ideal(c))))
        deeadeez = self.dee_activation_dee_z(self.output_layer)
        return deeadeez * deeC0deea

    def explicit_show_last_layer_activation(self,single_symbol_character_tuple):
        s,c = single_symbol_character_tuple
        print(c)
        self.one_shot(s)
        #self.dump_network()
        for j in range((np.shape(self.a[self.output_layer]))[0]):
            print('a_'+str(j)+'('+str(self.output_layer)+')='+str(self.a[self.output_layer].item(j)))

    def mnielsen_style_output_error(self,single_symbol_character_tuple):
        s,c = single_symbol_character_tuple
        activ_here = self.one_shot(s)
        dee_C_dee_a = 2*(np.square(activ_here - self.ideal(c)))
        dee_a_dee_z = self.dee_activation_dee_z(activ_here)
        delta_L = dee_C_dee_a * dee_a_dee_z
        print('')
        print(dee_C_dee_a)
        print(dee_a_dee_z)
        print(delta_L)
        return(delta_L)

    ### "node values" of Lague dee_a_dee_z * dee_C_dee_a



    def dump_network(self):
        for i,l in enumerate(self.a):
            print('\n#### layer '+str(i))
            print('##### weights')
            print(str(self.W[i]))
            print('##### bias')
            print(str(self.b[i]))
            print('##### weighted sums')
            print(str(self.z[i]))
            print('##### activations')
            print(str(self.a[i]))


class Handwritten:
    """Loading and manipulating handwritten symbols"""
    # Data and file format info from http://yann.lecun.com/exdb/mnist/
    bytes_per_label = 1
    bytes_per_pixel = 1

    def __init__(self,images_filename,labels_filename=None):
        self.dataset = []
        self.row_pix = None
        self.col_pix = None
        label_array = []
        if labels_filename:
            print('opening labels filename ' + labels_filename)
            with open(labels_filename, mode='rb') as labels:
                magic_number = int.from_bytes(labels.read(4),'big')
                print('magic number ',str(magic_number))
                if magic_number != 2049:
                    print('### labels file '+labels_filename+' has magic number '+str(magic_number)+' instead of 2051')
                    return
                number_of_labels = int.from_bytes(labels.read(4),'big')
                size = Handwritten.bytes_per_label
                for image in range(number_of_labels):
                    label_array.append(int.from_bytes(labels.read(size),'big'))
                    #print(str(label_array[-1]))
                if number_of_labels != len(label_array):
                    print('### the proper number of labels could not be extracted from labels file '+labels_filename+', '+str(len(label_array))+' vs '+number_of_labels)
                    return
        with open(images_filename, mode='rb') as images:
            magic_number = int.from_bytes(images.read(4),'big')
            print('magic number ',str(magic_number))
            if magic_number != 2051:
                print('### images file '+images_filename+' has magic number '+str(magic_number)+' instead of 2051')
                return
            number_of_images = int.from_bytes(images.read(4),'big')
            if label_array and (number_of_images != number_of_labels):
                print('### the number of images doesnt match the number of labels '+number_of_images+' vs '+number_of_labels)
                return
            self.row_pix = int.from_bytes(images.read(4),'big')
            self.col_pix = int.from_bytes(images.read(4),'big')
            size = self.row_pix * self.col_pix * Handwritten.bytes_per_pixel
            for image in range(100): #range(number_of_images):
                s = np.atleast_2d(np.array([x for x in images.read(size)])).T
                c = None if not label_array else label_array[image]
                self.dataset.append((s,c))
    
    def dump(self,range=None):
        index_pad = (len(str(len(self.dataset)-1)))
        for i, (s,c) in enumerate(self.dataset):
            index_tag = f'{i:{index_pad}}'
            s = np.right_shift(np.reshape(s,(self.row_pix,self.col_pix)),5)
            s = '\n'+np.array2string(s)
            s = s.replace('0',' ')
            k = s.rfind('\n [')
            s = s[:k] + '\n'+index_tag+': ( [' + s[k+3:]
            #before,after = s.split('\n [',maxsplit=1)
            #s = before + '\n'+index_tag+': ( [' + after
            s = s.replace('\n [','\n'+index_pad*' '+'    [')
            s = s.replace('\n[[','\n'+index_pad*' '+'   [[')
            print(s+', '+str(c)+') ')

    def show_training_pair(self,d):
        index_pad = 2
        i = 1
        if True:
            (s,c) = d
            index_tag = f'{i:{index_pad}}'
            s = np.right_shift(np.reshape(s,(self.row_pix,self.col_pix)),5)
            s = '\n'+np.array2string(s)
            s = s.replace('0',' ')
            k = s.rfind('\n [')
            s = s[:k] + '\n'+index_tag+': ( [' + s[k+3:]
            #before,after = s.split('\n [',maxsplit=1)
            #s = before + '\n'+index_tag+': ( [' + after
            s = s.replace('\n [','\n'+index_pad*' '+'    [')
            s = s.replace('\n[[','\n'+index_pad*' '+'   [[')
            print(s+', '+str(c)+') ')



#np.random.random_sample((9,1))
p = Slinkytron([3*3,3,3,4])
p.slantify()
p.one_shot(np.random.random_sample((9,1)))

# print('')
# test = 10*np.random.random_sample((9,1))-5
# print(test)
# test = p.activation(test)
# print(test)
# test = p.dee_activation_dee_z(test)
# print(test)
# test = p.dee_activation_dee_z(np.array([[-2,-1,0,1,2]]))
# print(test)

#p = Slinkytron([28*28,16,16,10])
#p = Slinkytron([5,4,4,2])
#p.dump_network()
print('')
#p.one_shot(np.random.random_sample((785,2)))
#p.one_shot(np.random.random_sample((784,1)))

#print(p.one_shot(np.ones((3*3,1))*0.5))
#p.slantify()
#print(p.one_shot(np.ones((3*3,1))*3))
#print(p.one_shot(np.random.random_sample((3*3,1))))

#print(p.perfect_activation(9).T)
#p.dump_network()

#digits = Handwritten('train-images.dat')
####digits = Handwritten('train-images.dat',labels_filename='train-labels.dat')

####for d in digits.dataset:
    ####print('')
    ####digits.show_training_pair(d)
    # cost = p.single_symbol_cost(d)
    # print(cost)
    # dee_C_dee_a = p.dee_single_symbol_cost_dee_a(d)
    # print(dee_C_dee_a)
    # dee_C_dee_z = p.dee_single_symbol_cost_dee_z(d)
    # print(dee_C_dee_z)
    #p.dee_single_symbol_cost_dee_w(d)
    ####p.explicit_show_last_layer_activation(d)

#c = p.cost_function(digits.dataset)
#print(c)

#digits.dump()
