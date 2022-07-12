import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# scipy.integrate.ode

def integrate_variational(x0, tf, f, df, ddf, dim, **kwargs):

    def variational_eq(x, Y):
        Y_dot = np.matmul(df(x), Y)
        return Y_dot.flatten()

    def full_problem(t,long_x):
        x = long_x[0:dim]
        y = long_x[dim:]
        x_dot = f(x)
        squareY = np.reshape(y, [dim, dim])
        y_dot = variational_eq(x, squareY)

        long_y = np.zeros([dim + dim * dim])
        long_y[0:dim] = x_dot
        long_y[dim:] = y_dot
        return long_y

    def der_full_problem(t,long_x):
        x = long_x[0:dim]
        y = long_x[dim:]
        DF = np.zeros([dim + dim * dim, dim + dim * dim])
        DF[0:dim, 0:dim] = df(x)
        DF[dim:-1, 0:dim] = np.reshape(np.einsum('lki,kj->lkj', ddf(x), y), [dim, dim * dim])
        DF_temp = np.zeros([dim ** 2, dim ** 2])
        df_x = df(x)
        for l in range(dim):
            for m in range(dim):
                row = l * dim + m
                for i in range(dim):
                    column = i * dim + m
                    DF_temp[row, column] = df_x[i, l]
        DF[dim:, dim:] = DF_temp
        return DF

    y0 = np.identity(dim).flatten()
    initial_cond = np.zeros([dim + dim * dim])
    initial_cond[0:dim] = x0
    initial_cond[dim:] = y0
    results = solve_ivp(full_problem, (0, tf), initial_cond, jac=der_full_problem,
                        **kwargs)
    return results


def extract_lyap_exp(results, dim):
    T = results.t[-1]
    long_x = results.y[:, -1]
    y = long_x[dim:]
    Y = np.reshape(y, [dim, dim])
    A = Y.T * Y
    w, v = np.linalg.eig(A)
    lyap = np.log(np.max(w))/(2*T)
    return lyap


def RNN_rhs(W):
    def rhs(x):
        return np.tanh(W.dot(x))
    def der_rhs(x):
        y = W * x
        der = np.diag(1 - np.tanh(y)** 2) * W
        return der
    def dd_rhs(x):
        y = W * x
        v = 2 * np.tanh(y) * (1 - np.tanh(y) * np.tanh(y))
        dxxfv = - np.eisum('ik,ij,i->ikj', W, W, v)
        return dxxfv
    return rhs, der_rhs, dd_rhs


def detect_lyap_on_RNNmat(W, time=100):
    dim = W.shape[0]
    x0 = np.random.random([dim])
    rnn_f, rnn_df, rnn_ddf = RNN_rhs(W)
    res = integrate_variational(x0, time, rnn_f, rnn_df, rnn_ddf, dim)
    lyap_exp = extract_lyap_exp(res, dim)
    return lyap_exp


def debug_tests():
    def f(x):
        return np.array([x[1], - x[0]])
    def df(x):
        return np.array([[0,1],[-1,0]])
    def ddf(x):
        return np.zeros([2,2,2])

    x0 = np.array([2,3])
    tf = 4
    res = integrate_variational(x0, tf, f, df, ddf, 2)
    print(res)
    lyap_exp = extract_lyap_exp(res, 2)
    print(lyap_exp)
    W_mat = np.array([[1,2],[3,-1]])
    rnn_f, rnn_df, rnn_ddf = RNN_rhs(W_mat)
    res = integrate_variational(x0, tf, rnn_f, rnn_df, rnn_ddf, 2)
    print(res)
    lyap_exp = extract_lyap_exp(res, 2)
    print(lyap_exp)
    lyap_2 = detect_lyap_on_RNNmat(W_mat)
    print(lyap_2)

    iters = 400
    lyap_vec = np.zeros([iters])
    for i in range(iters):
        dim = 4
        W_mat = np.random.random([dim, dim])
        lyap_vec[i] = detect_lyap_on_RNNmat(W_mat)
    plt.hist(lyap_vec, bins=50)
    plt.title('100')
    plt.show()

    return


if __name__ == "__main__":
    iters = 200
    time = 300
    for power in range(0, -5, -1):
        epsilon = 10**power
        lyap_vec = np.zeros([iters])
        for i in range(iters):
            dim = 40
            W_mat = np.random.random([dim, dim])
            perturbation = epsilon*np.random.random([dim, dim])
            Id = np.identity(dim)
            gamma = 0.001
            W_mat = W_mat.T - W_mat + gamma*Id + perturbation
            lyap_vec[i] = detect_lyap_on_RNNmat(W_mat, time=time)
        plt.hist(lyap_vec, bins=50)
        plt.title('time = 30,'+str(epsilon)+' perturbation')
        plt.ylabel('likelihood('+str(iters)+')')
        plt.xlabel('approx lyap exp')
        name = './perturbation_figures/power'+str(power)+'time'+str(time)
        plt.savefig(name)
        plt.show()
