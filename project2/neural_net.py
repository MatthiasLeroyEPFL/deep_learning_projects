from torch import LongTensor, FloatTensor
import math


class Module(object):  
    def forward(self , *x):
        raise NotImplementedError
    
    def backward(self , *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []
    
    def step(self, eta):
        pass
    
    def grad_zero(self):
        pass



class Sequential(Module):
    '''
    Sequential Module
    '''
    
    
    def __init__(self, *args):
        '''
        Keeps in memory every module passed in argument
        '''
        super(Sequential, self).__init__()
        self.module_array = []
        for arg in args:
            self.module_array.append(arg)
    
    
    def forward(self, *x):
        '''
        Applies the forward step to each module in the order of the list
        '''
        input_ = x[0]
        for module in self.module_array:           
            input_ = module.forward(input_)
        return input_
    
    
    def backward(self, *gradwrtoutput):
        '''
        Applies the backward step to each module in the inverse order of the list
        '''
        input_ = gradwrtoutput[0]
        for module in self.module_array[::-1]:
            input_ = module.backward(input_)            
        return input_
    
    
    def param(self):
        '''
        Returns every parameters
        '''
        parameters = []
        for module in self.module_array:
            parameters.extend(module.param())
        return parameters
    
    
    def step(self, eta):
        '''
        Updates the weight and bias for every modules
        '''
        for module in self.module_array:
            module.step(eta)
    
    def grad_zero(self):
        '''
        Resets the gradients to zero for each module
        '''
        for module in self.module_array:
            module.grad_zero()
        



class Linear(Module):
    '''
    Linear Module
    '''
    
    def __init__(self, in_features, out_features):
        '''
        Initializes weight, bias, gradient of weight and gradient of bias
        '''
        super(Linear, self).__init__()
        self.weight = FloatTensor(out_features, in_features)
        self.bias = FloatTensor(out_features).view(-1,1)
        self.reset_parameters()
        self.bias_grad = FloatTensor(self.bias.size()).zero_()
        self.weight_grad = FloatTensor(self.weight.size()).zero_()
        self.previous_x = None        
    
    
    def reset_parameters(self):
        '''
        Initializes weight and bias with uniform law. Taken from Lecture 5 of Deep Learning
        '''
        std = 1 / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-std, std)
        self.bias.uniform_(-std, std)
    
    
    def forward(self, x):
        '''
        Computes forward step of the Linear module
        '''
        self.previous_x = x
        return self.weight.matmul(x) + self.bias
    
    
    def backward(self, *gradwrtoutput):
        '''
        Computes backward step of the Linear module
        '''
        self.bias_grad.add_(gradwrtoutput[0].sum(1))
        self.weight_grad.add_(gradwrtoutput[0].matmul(self.previous_x.t()))
        
        return self.weight.t().matmul(gradwrtoutput[0])
    
    
    def step(self, eta):
        '''
        Updates the weight and bias after gradient step
        '''
        self.weight = self.weight - eta * self.weight_grad
        self.bias = self.bias - eta * self.bias_grad
    
    
    def grad_zero(self):
        '''
        Resets the gradients to zero after gradient step
        '''
        self.bias_grad.zero_()
        self.weight_grad.zero_()
    
    
    def param(self):
        '''
        Return the weight and the bias
        '''
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]   



class ReLU(Module):
    '''
    Rectified linear unit function module
    '''
    
    def __init__(self):
        super(ReLU, self).__init__()
        self.previous_x = None
    
    
    def forward(self, x):
        '''
        Computes forward step of the ReLU module
        '''
        self.previous_x = x
        x[x<=0] = 0
        return x
    
    
    def backward(self, *gradwrtoutput):
        '''
        Computes backward step of the ReLU module
        '''
        return gradwrtoutput[0] * self.dRelu(self.previous_x)
    
    
    def dRelu(self, x):
        '''
        Computes the derivative of the ReLU function
        '''
        x[x>0] = 1
        x[x<0] = 0
        return x
        

class Tanh(Module):
    '''
    Tanh function module
    '''
    def __init__(self):
        super(Tanh, self).__init__()
        self.previous_x = None
    
    
    def forward(self, x):
        '''
        Computes forward step of the Tanh module
        '''
        self.previous_x = x
        return x.tanh()
    
    
    def backward(self, *gradwrtoutput):
        '''
        Computes backward step of the Tanh module
        '''
        return gradwrtoutput[0] * self.dTanh(self.previous_x)
    
    
    def dTanh(self, x):
        '''
        Computes the derivative of the Tanh function
        '''
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
        


class MSELoss(Module):
    '''
    Mean square error loss module
    '''
    def __init__(self):
        super(MSELoss, self).__init__()
       
    def forward(self, *x):
        '''
        Computes the MSE loss between prediction and target
        ''' 
        return (x[0] - x[1]).pow(2).sum()
    
    
    def backward(self, *gradwrtoutput):
        '''
        Computes the gradient of the loss
        '''
        return 2 * (gradwrtoutput[0] - gradwrtoutput[1])
