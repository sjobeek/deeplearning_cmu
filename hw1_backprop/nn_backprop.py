
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


IMG_DIM = 28

def tomat(vec):
    img_dim = int(np.sqrt(len(vec)))
    return vec.reshape((img_dim,img_dim))

def plot_1(img_vec):
    """img_mat must be single vector-representation of image to plot"""
    if len(img_vec.shape) <= 1:
        img_mat = tomat(img_vec)
    else:
        img_mat = img_vec # It was a matrix already
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img_mat, cmap=mpl.cm.binary)
    ax.axis('off')
    plt.show()

def plot_100(img_vec_array):
    first_100 = img_vec_array[:100]
    image_mat_10x10 = np.zeros((IMG_DIM*10, IMG_DIM*10))
    for x in range(10):
        for y in range(10):
            # Replace sub-matrix with appropriate values
            image_mat_10x10[IMG_DIM*y : IMG_DIM*y+IMG_DIM,
                            IMG_DIM*x : IMG_DIM*x+IMG_DIM] = tomat(first_100[10*y + x])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image_mat_10x10, cmap=mpl.cm.binary)
    ax.axis('off')
    plt.show()
    return fig

def sig(x):                                        
    return 1 / (1 + np.exp(-x))

def w_scale(fan_in=100, fan_out=100):
    return 4*np.sqrt(6/(fan_in+fan_out))

def softmax(x):
    """Numerically stable version..."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def d_sig(x):
    return sig(x)*(1-sig(x))

def binary_vec_plot(vec, title=""):
    plt.matshow(np.matrix(vec), cmap=mpl.cm.binary)
    plt.title(title)
    
def grad_vec_plot(vec, title=""):
    plt.matshow(np.matrix(vec), cmap=mpl.cm.RdBu)
    plt.title(title)



def train_singlelayer(epochs=200, rate=0.1, momentum=0.0, dropout=0.0, L2_reg=0.0,
                      train_filepath='./data/digitstrain.txt',
                      test_filepath='./data/digitstest.txt',
                      validation_filepath='./data/digitsvalid.txt',
                      seed=42, n_hidden=100, validation=False):

    alldata_train = np.loadtxt(train_filepath, delimiter=',')
    alldata_test = np.loadtxt(test_filepath, delimiter=',')

    momentum = 1 - momentum # Make sure 0 = no momentum (was opposite)
    np.random.seed(seed)
    np.random.shuffle(alldata_train)
    np.random.shuffle(alldata_test)

    y_train = alldata_train[:,-1].astype(int)
    digits_train = alldata_train[:,:-1]

    y_test = alldata_test[:,-1].astype(int)
    digits_test = alldata_test[:,:-1]

    if validation==True:
        alldata_valid = np.loadtxt(validation_filepath, delimiter=',')
        np.random.shuffle(alldata_valid)
        y_test = alldata_valid[:,-1].astype(int)
        digits_test = alldata_valid[:,:-1]



    outputs = []

    ce_errors_train = [] # Cross entropy errors for test
    ce_test_per_epoch = []
    ce_errors_test = []
    ce_train_per_epoch = []
    class_errors_test = []
    class_test_per_epoch = []
    class_errors_train = []
    class_train_per_epoch = []

    b1 = np.zeros(n_hidden)
    b2 = np.zeros(10)
    W1 = np.random.uniform(w_scale(784, n_hidden), -w_scale(784, n_hidden), size=(n_hidden,784))
    Wout = np.random.uniform(w_scale(n_hidden, 10), -w_scale(n_hidden, 10), size=(10,n_hidden))
    PreActOut_grad = 0
    b2_grad = 0
    Wout_grad = 0
    h1_grad = 0
    PreAct1_grad = 0
    b1_grad = 0
    W1_grad = 0



    for epoch in range(epochs):
        # Evaluate Training Data
        for n in range(len(digits_test)):
            ###### Forward Pass
            h0 = digits_test[n]
            # Hidden Layer 1
            PreAct1 = W1 @ h0 + b1
            h1 = sig(PreAct1)
            # Output Layer

            PreActOut = Wout @ h1 + b2
            output_test = softmax(PreActOut)

            # True value
            indicator_vector_test = np.zeros(10)
            indicator_vector_test[y_test[n]] = 1  # Set  
            outputs.append(indicator_vector_test)

            ce_errors_test.append(np.sum(-np.multiply(np.log(output_test),indicator_vector_test)))
            if np.argmax(output_test) == y_test[n]:
                class_errors_test.append(0)
            else:
                class_errors_test.append(1)




        for n in range(len(digits_train)):

            dropout_mask = np.random.binomial(1,(1-dropout), n_hidden)
            
            ###### Forward Pass
            h0 = digits_train[n]
            # Hidden Layer 1
            PreAct1 = W1 @ h0 + b1
            h1 = sig(PreAct1) * dropout_mask
            # Output Layer

            PreActOut = Wout @ h1 + b2
            output = softmax(PreActOut)

            # Backward Pass
            # Output of network 

            # True value
            indicator_vector = np.zeros(10)
            indicator_vector[y_train[n]] = 1  # Set  
            outputs.append(indicator_vector)


            ce_errors_train.append(np.sum(-np.multiply(np.log(output),indicator_vector)))
            if np.argmax(output) == y_train[n]:
                class_errors_train.append(0)
            else:
                class_errors_train.append(1)

            #### Start output layer
            # Resulting gradient at lowest level (inverse sign from notes)
            PreActOut_grad = (momentum * (indicator_vector - output) + (1- momentum)*PreActOut_grad + 
                             L2_reg * np.sum(W1.flatten()*W1.flatten()))
              # [10]
            # How much cost will change per pre-activation (at X)

            # connection weight deltas
            b2_grad = momentum * PreActOut_grad + (1- momentum)*b2_grad
            Wout_grad = momentum * np.array(np.matrix(PreActOut_grad).T @ np.matrix(h1)) + (1- momentum)*Wout_grad     # [10,100]

            # Apply updates
            b2 = b2 + rate*b2_grad
            Wout = Wout + rate*Wout_grad
            #### End Output Layer

            #### Start hidden layer 1
            # How much cost will change per hidden unit activation value
            h1_grad = momentum * Wout.T @ PreActOut_grad  + (1- momentum)* h1_grad # [100]
            PreAct1_grad = momentum * np.multiply(h1_grad,d_sig(PreAct1)) + (1- momentum)*PreAct1_grad # Element-wise multiplication

            b1_grad = momentum * PreAct1_grad + (1- momentum)*b1_grad
            W1_grad = momentum * np.matrix(PreAct1_grad).T @ np.matrix(h0) + (1- momentum)*W1_grad
            # Apply updates
            b1 = b1 + rate*b1_grad
            W1 = W1 + rate*np.array(W1_grad)
            #### End hidden layer 1

        ce_train_per_epoch.append(np.average(ce_errors_train))
        ce_errors_train = []        
        ce_test_per_epoch.append(np.average(ce_errors_test))
        ce_errors_test = []
        class_test_per_epoch.append(np.average(class_errors_test))
        class_errors_test = []
        class_train_per_epoch.append(np.average(class_errors_train))
        class_errors_train = []


    result = {'output': output,
              'W1': W1,
              'Wout': Wout,
              'ce_test_per_epoch': ce_test_per_epoch,
              'ce_train_per_epoch': ce_train_per_epoch,
              'class_test_per_epoch': class_test_per_epoch,
              'class_train_per_epoch': class_train_per_epoch}

    return result


def train_twolayer(epochs=200, rate=0.1, momentum=0.0, dropout=0,
                      train_filepath='./data/digitstrain.txt',
                      test_filepath='./data/digitstest.txt',
                      validation_filepath='./data/digitsvalid.txt',
                      seed=42, validation=False):

    alldata_train = np.loadtxt(train_filepath, delimiter=',')
    alldata_test = np.loadtxt(test_filepath, delimiter=',')

    momentum = 1 - momentum # Make sure 0 = no momentum (was opposite)
    np.random.seed(seed)
    np.random.shuffle(alldata_train)
    np.random.shuffle(alldata_test)

    y_train = alldata_train[:,-1].astype(int)
    digits_train = alldata_train[:,:-1]

    y_test = alldata_test[:,-1].astype(int)
    digits_test = alldata_test[:,:-1]

    if validation==True:
        alldata_valid = np.loadtxt(validation_filepath, delimiter=',')
        np.random.shuffle(alldata_valid)
        y_test = alldata_valid[:,-1].astype(int)
        digits_test = alldata_valid[:,:-1]


    outputs = []

    ce_errors_train = [] # Cross entropy errors for test
    ce_test_per_epoch = []
    ce_errors_test = []
    ce_train_per_epoch = []
    class_errors_test = []
    class_test_per_epoch = []
    class_errors_train = []
    class_train_per_epoch = []

    b1 = np.zeros(100)
    b2 = np.zeros(100)
    b3 = np.zeros(10)
    W1 = np.random.uniform(w_scale(784, 100), -w_scale(784, 100), size=(100,784))
    W2 = np.random.uniform(w_scale(100, 100), -w_scale(100, 100), size=(100,100))
    Wout = np.random.uniform(w_scale(100, 10), -w_scale(100, 10), size=(10,100))
    PreActOut_grad = 0

    h1_grad = 0
    h2_grad = 0
    PreAct1_grad = 0
    PreAct2_grad = 0
    b1_grad = 0
    b2_grad = 0
    b3_grad = 0
    W1_grad = 0
    W2_grad = 0
    Wout_grad = 0


    for epoch in range(epochs):
        # Evaluate Training Data
        for n in range(len(digits_test)):
            ###### Forward Pass
            h0 = digits_test[n]
            # Hidden Layer 1
            PreAct1 = W1 @ h0 + b1
            h1 = sig(PreAct1)

            # Hidden Layer 2
            PreAct2 = W2 @ h1 + b2
            h2 = sig(PreAct2)

            # Output Layer
            PreActOut = Wout @ h2 + b3
            output_test = softmax(PreActOut)

            # True value
            indicator_vector_test = np.zeros(10)
            indicator_vector_test[y_test[n]] = 1  # Set  
            outputs.append(indicator_vector_test)

            ce_errors_test.append(np.sum(-np.multiply(np.log(output_test),indicator_vector_test)))
            if np.argmax(output_test) == y_test[n]:
                class_errors_test.append(0)
            else:
                class_errors_test.append(1)




        for n in range(len(digits_train)):
            ###### Forward Pass
            h0 = digits_train[n]
            # Hidden Layer 1
            PreAct1 = W1 @ h0 + b1
            h1 = sig(PreAct1)

            # Hidden Layer 2
            PreAct2 = W2 @ h1 + b2
            h2 = sig(PreAct2)

            # Output Layer
            PreActOut = Wout @ h2 + b3
            output = softmax(PreActOut)

            # Backward Pass
            # Output of network 

            # True value
            indicator_vector = np.zeros(10)
            indicator_vector[y_train[n]] = 1  # Set  
            outputs.append(indicator_vector)


            ce_errors_train.append(np.sum(-np.multiply(np.log(output),indicator_vector)))
            if np.argmax(output) == y_train[n]:
                class_errors_train.append(0)
            else:
                class_errors_train.append(1)

            #### Start output layer
            # Resulting gradient at lowest level (inverse sign from notes)
            PreActOut_grad = momentum * (indicator_vector - output) + (1- momentum)*PreActOut_grad
              # [10]
            # How much cost will change per pre-activation (at X)

            # connection weight deltas
            b3_grad = momentum * PreActOut_grad + (1- momentum)*b3_grad
            Wout_grad = momentum * np.array(np.matrix(PreActOut_grad).T @ np.matrix(h1)) + (1- momentum)*Wout_grad     # [10,100]

            # Apply updates
            b3 = b3 + rate*b3_grad
            Wout = Wout + rate*Wout_grad
            #### End Output Layer

            #### Start hidden layer 2
            # How much cost will change per hidden unit activation value
            h2_grad = momentum * Wout.T @ PreActOut_grad  + (1- momentum)* h2_grad # [100]
            PreAct2_grad = momentum * np.multiply(h2_grad,d_sig(PreAct2)) + (1- momentum)*PreAct2_grad # Element-wise multiplication

            b2_grad = momentum * PreAct2_grad + (1- momentum)*b2_grad
            W2_grad = momentum * np.matrix(PreAct2_grad).T @ np.matrix(h1) + (1- momentum)*W2_grad
            # Apply updates
            b2 = b2 + rate*b2_grad
            W2 = W2 + rate*np.array(W2_grad)
            #### End hidden layer 1


            #### Start hidden layer 1
            # How much cost will change per hidden unit activation value
            h1_grad = momentum * W2.T @ PreAct2_grad  + (1- momentum)* h1_grad # [100]
            PreAct1_grad = momentum * np.multiply(h1_grad,d_sig(PreAct1)) + (1- momentum)*PreAct1_grad # Element-wise multiplication

            b1_grad = momentum * PreAct1_grad + (1- momentum)*b1_grad
            W1_grad = momentum * np.matrix(PreAct1_grad).T @ np.matrix(h0) + (1- momentum)*W1_grad
            # Apply updates
            b1 = b1 + rate*b1_grad
            W1 = W1 + rate*np.array(W1_grad)
            #### End hidden layer 1

        ce_train_per_epoch.append(np.average(ce_errors_train))
        ce_errors_train = []        
        ce_test_per_epoch.append(np.average(ce_errors_test))
        ce_errors_test = []
        class_test_per_epoch.append(np.average(class_errors_test))
        class_errors_test = []
        class_train_per_epoch.append(np.average(class_errors_train))
        class_errors_train = []


    result = {'output': output,
              'W1': W1,
              'Wout': Wout,
              'ce_test_per_epoch': ce_test_per_epoch,
              'ce_train_per_epoch': ce_train_per_epoch,
              'class_test_per_epoch': class_test_per_epoch,
              'class_train_per_epoch': class_train_per_epoch}

    return result

if __name__ == "__main__":
    result_dict = train_singlelayer(epochs=30, rate=0.05)
    plt.plot(result_dict['ce_test_per_epoch'], c='r')
    plt.plot(result_dict['ce_train_per_epoch'], c='b')
    plt.plot(result_dict['class_test_per_epoch'], c='g')
    plt.plot(result_dict['class_train_per_epoch'], c='k')
    print('Min Classification Error (test):', 
          min(result_dict['class_test_per_epoch']))
    plt.title("Validation Error Rate vs Epoch")
    plt.show()
    fig = plot_100(result_dict['W1'])
    fig.show()