import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    return np.sin(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_RNN(U, V, W, x, T):
    hidden_dim = W.shape[0]
    prev_s = np.zeros((hidden_dim, 1))
    # Forward pass
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s
    return prev_s


def compute_loss(U, V, W, X, Y, T):
    loss = 0
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]  # get input, output values of each record
        prev_s = np.zeros((hidden_dim,
                           1))  # here, prev-s is the value of the previous activation of hidden layer; which is initialized as all zeroes
        for t in range(T):
            new_input = np.zeros(x.shape)  # we then do a forward pass for every timestep in the sequence
            new_input[t] = x[t]  # for this, we define a single input for that timestep
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s

        # calculate error
        loss_per_record = (y - mulv) ** 2 / 2
        loss += loss_per_record
    loss = loss / float(Y.shape[0])
    return loss


def update_weights(U, V, W, x, y, T, learning_rate):
    # train model

    layers = []
    prev_s = np.zeros((hidden_dim, 1))
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)

    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape)

    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape)

    # forward pass
    for t in range(T):
        new_input = np.zeros(x.shape)
        new_input[t] = x[t]
        mulu = np.dot(U, new_input)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        layers.append({'s': s, 'prev_s': prev_s})
        prev_s = s
    # derivative of pred
    dmulv = (mulv - y)

    # backward pass
    for t in range(T):
        dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
        dsv = np.dot(np.transpose(V), dmulv)

        ds = dsv
        dadd = add * (1 - add) * ds

        dmulw = dadd * np.ones_like(mulw)

        dprev_s = np.dot(np.transpose(W), dmulw)

        for j in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
            ds = dsv + dprev_s
            dadd = add * (1 - add) * ds

            dmulw = dadd * np.ones_like(mulw)
            dmulu = dadd * np.ones_like(mulu)

            dW_i = np.dot(W, layers[t]['prev_s'])
            dprev_s = np.dot(np.transpose(W), dmulw)

            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            dU_i = np.dot(U, new_input)
            dx = np.dot(np.transpose(U), dmulu)

            dU_t += dU_i
            dW_t += dW_i

        dV += dV_t
        dU += dU_t
        dW += dW_t

        if dU.max() > max_clip_value:
            dU[dU > max_clip_value] = max_clip_value
        if dV.max() > max_clip_value:
            dV[dV > max_clip_value] = max_clip_value
        if dW.max() > max_clip_value:
            dW[dW > max_clip_value] = max_clip_value

        if dU.min() < min_clip_value:
            dU[dU < min_clip_value] = min_clip_value
        if dV.min() < min_clip_value:
            dV[dV < min_clip_value] = min_clip_value
        if dW.min() < min_clip_value:
            dW[dW < min_clip_value] = min_clip_value

    # update
    U -= learning_rate * dU
    V -= learning_rate * dV
    W -= learning_rate * dW
    return U, V, W


# target result
data = [target_function(x) for x in range(200)]

length_sequence = 50
n_data = len(data) - 2*length_sequence
n_val_data = length_sequence

X_training = np.zeros([n_data, length_sequence, 1])
Y_training = np.zeros([n_data, 1, 1])
X_val = np.zeros([n_val_data, length_sequence,1])
Y_val = np.zeros([n_val_data, 1, 1])

for i in range(n_data):
    X_training[i, :, 0] = data[i:i+length_sequence]
    Y_training[i, 0] = data[i+length_sequence]
for i in range(n_val_data):
    X_val[i, :, 0] = data[n_data+i:n_data+i+length_sequence]
    Y_val[i, 0] = data[n_data + i + length_sequence]


learning_rate = 0.00015
nepoch = 25
T = 50                   # length of sequence
hidden_dim = 100
output_dim = 1

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))


for epoch in range(nepoch):
    # check loss on train
    loss = compute_loss(U, V, W, X_training, Y_training, T)
    val_loss = compute_loss(U, V, W, X_val, Y_val, T)
    print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)
    for i in range(Y_training.shape[0]):
        x, y = X_training[i], Y_training[i]
        U, V, W = update_weights(U, V, W,  x, y, T, learning_rate)


preds = []
for i in range(Y_training.shape[0]):
    x, y = X_training[i], Y_training[i]
    preds.append(evaluate_RNN(U, V, W, x, T))

preds = np.array(preds)
plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y_training[:, 0], 'r')
plt.show()