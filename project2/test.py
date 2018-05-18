from neural_net import *




def create_target(target):
    '''
    Creates target with two dimensions to match the output
    '''
    first_dim, second_dim = [], []
    for v in target:
        if v == 1:
            first_dim.append(-1)
            second_dim.append(1)
        else:
            first_dim.append(1)
            second_dim.append(-1)
    return LongTensor([first_dim, second_dim]).t()        


def generate_disc_set(nb):
    '''
    Creates input and target
    '''
    input_ = FloatTensor(nb, 2).uniform_(0, 1)
    target = (input_[:,0] - 0.5).pow(2) + (input_[:,1] - 0.5).pow(2) < math.pow(1 / math.sqrt(2 * math.pi), 2)
    return input_, target


train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)


def train_model(model, train_input, train_target, mini_batch_size=10, eta=1e-2, nb_epochs=200):
    mse_loss = MSELoss()
    
    for e in range(0, nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            
            train_input_t = train_input.narrow(0, b, mini_batch_size).t()
            train_target_t = train_target.narrow(0, b, mini_batch_size).t()
            
            output = model.forward(train_input_t)
            sum_loss += mse_loss.forward(output, train_target_t)
            grad_loss = mse_loss.backward(output, train_target_t)
            model.grad_zero()
            model.backward(grad_loss)
            model.step(eta)
        print('epoch {}: loss = {}'.format(e+1,sum_loss))


def compute_nb_errors(model, data_input, data_target, mini_batch_size=10):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        
        data_input_t = data_input.narrow(0, b, mini_batch_size).t()
        
        output = model.forward(data_input_t)
        _, predicted_classes = output.max(0)
        for k in range(0, mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors / data_input.size(0) * 100


model = Sequential(Linear(2,25),ReLU(), Linear(25,25), ReLU(), Linear(25,25), Tanh(), Linear(25,2))
train_model(model, train_input, create_target(train_target).float(), eta=1e-3, nb_epochs=300)


train_accuracy = 100 - compute_nb_errors(model, train_input, train_target)
print('Train accuracy: ', train_accuracy )
test_accuracy = 100 - compute_nb_errors(model, test_input, test_target)
print('Test accuracy: ', test_accuracy )



