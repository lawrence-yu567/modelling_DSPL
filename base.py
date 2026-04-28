import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro as npr
import numpyro.distributions as dist
import corner
from numpyro.infer import MCMC, NUTS
from scipy.special import roots_legendre
from jax.scipy.stats import gaussian_kde
from numpyro.handlers import condition # for fixing parameters in model



#beta = 0.706
# Collett & Auger 2014, for Jackpot Lens


#z_l = 0.222
#z_s1 = 0.609
#z_s2 = 2.035


# from Dan
def nth_order_quad(n=20):
    xval, weights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    weights = weights.reshape(-1, 1)

    def integrate(func, a, b, args=()):
        # Integrate function with args from a to b
        return 0.5 * (b - a) * jnp.sum(
            weights * func(0.5 * ((b - a) * xval + (b + a)), *args),
            axis=0
        )

    return integrate
quad = nth_order_quad() # jax compatible quad function for integration



#functions for calculating beta
def E(z, w_0, w_a, OmM):
    return jnp.sqrt(OmM*(1+z)**3 + (1-OmM)*(1+z)**(3*(1+w_0+w_a))*jnp.exp((-3*w_a*z)/(1+z)))

def integral(z_i, z_j, w_0, w_a, OmM):
    integrant = lambda z: 1.0 / E(z, w_0, w_a, OmM)
    return quad(integrant, z_i, z_j)

def b(z_l, z_s1, z_s2, w_0, w_a, OmM):
    return (integral(z_l, z_s1, w_0, w_a, OmM)*integral(0, z_s2, w_0, w_a, OmM)/
            (integral(z_l, z_s2, w_0, w_a, OmM)*integral(0, z_s1, w_0, w_a, OmM))
    )

#for reading DESI data
def read_txt(filename, column):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split()
            data.append(values[column-1])
    return np.array(data[1:], dtype=float)


#likelihood function for MCMC, 
def model():
    w_0 = npr.sample('w_0', dist.Normal(-0.55, 3.9))
    w_a = npr.sample('w_a', dist.Uniform(-3, -1.32))
    OmM = npr.sample('OmM', dist.Normal(0.344, 0.47))

    z_l = npr.sample("z_l", dist.Uniform(0.2, 2))
    z_s1 = npr.sample("z_s1", dist.Uniform(z_l, 4))
    z_s2 = npr.sample("z_s2", dist.Uniform(z_s1, 6))
    measured_error_beta = npr.sample("error",dist.Uniform(0.001, 10))

    model_beta = npr.deterministic("b", b(z_l, z_s1, z_s2, w_0, w_a, OmM))

    measured_beta = b(z_l, z_s1, z_s2, -1.0, 0.0, 0.3) # beta at LCDM parameters
    #measured_error_beta = measured_beta * 0.005 # 0.5% error
    
    npr.sample('likelihood',dist.Normal(model_beta, measured_error_beta),obs=measured_beta)


def joint_model(z_l, z_s1, z_s2):
    w_0 = npr.sample('w_0', dist.Uniform(-3, 1))
    w_a = npr.sample('w_a', dist.Uniform(-3, 2))
    OmM = npr.sample('OmM', dist.Uniform(0, 1))
    #measured_error_beta = npr.sample("error",dist.Uniform(0.001, 10))

    model_beta = npr.deterministic("b", b(z_l, z_s1, z_s2, w_0, w_a, OmM))

    w_0_desi = read_txt("chain.1.txt", 3)
    w_a_desi = read_txt("chain.1.txt", 4)
    OmM_desi = read_txt("chain.1.txt", 6)

    desi_kde = gaussian_kde(jnp.vstack([OmM_desi, w_0_desi, w_a_desi]))
    npr.deterministic("desi_likelihood", jnp.log(desi_kde(jnp.array([OmM, w_0, w_a]))))
    #npr.factor("desi_likelihood", jnp.log(desi_kde([0.3, -1.0, 0.0])))

    measured_beta = b(z_l, z_s1, z_s2, -1.0, 0.0, 0.3) # beta at LCDM parameters
    measured_error_beta = measured_beta * 0.001 # 0.5% error, subject to change
    
    npr.sample('likelihood',dist.Normal(model_beta, measured_error_beta),obs=measured_beta)


#single_variable(3, 4, 0.1)

#this gives some interesting plots
#double_variable(2.0, 3.0, 0.1, 5, 6, 0.1, 2.0)