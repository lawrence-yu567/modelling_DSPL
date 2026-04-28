from base import *
from pathlib import Path

w_0_desi = read_txt("chain.1.txt", 3)
w_a_desi = read_txt("chain.1.txt", 4)
OmM_desi = read_txt("chain.1.txt", 6)

w_a_desi_std = np.std(w_a_desi)
w_0_desi_std = np.std(w_0_desi)

axis_range = [(0.0, 1.0), (-3.0, 1.0), (-3.0, 2.0)]


def variable_lens():
    #z_l = np.arange(0.1, 2, 0.1)
    z_l = 0.2
    z_s1 = 0.25
    z_s2_range = np.arange(0.3, 2.5, 0.1)

    #model_cond=condition(model, {'w_a': jnp.array(0.0)})

    w_0_desi = read_txt("chain.1.txt", 3)
    w_a_desi = read_txt("chain.1.txt", 4)
    OmM_desi = read_txt("chain.1.txt", 6)

    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=100000, num_chains=1)
    key = jax.random.PRNGKey(100)

    for z_s2 in z_s2_range:

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()
    
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        joint = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "black",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            weights=desi_likelihood
        )

        desi = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "grey",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            fill_contours = False,
            plot_datapoints = False,
            fig = joint
        )

        fig_final = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            fig = desi,
            truths=[0.3, -1.0, 0.0]
        )

        plt.suptitle(f'z_l={z_l:.2f}  z_s1={z_s1:.2f}  z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"corner_plot_z_s2_{z_s2:.2f}.png")
        plt.close()    


def single_variable(a, b, step):

    z_l = 2
    z_s1 = 2.1
    z_s2 = np.arange(a, b, step)

    #model_cond=condition(model, {'w_a': jnp.array(0.0)})

    name = NUTS(model)
    mcmc = MCMC(name, num_warmup=1000, num_samples=10000, num_chains=1)
    key = jax.random.PRNGKey(100)


    w_0_desi = read_txt("chain.1.txt", 3)
    w_a_desi = read_txt("chain.1.txt", 4)
    OmM_desi = read_txt("chain.1.txt", 6)

    print(w_0_desi)
    print(w_a_desi)
    print(OmM_desi)


    for i in z_s2:
        
        mcmc.run(key, z_l, z_s1, i)
        #mcmc.print_summary()
    

        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']

        figure = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "red",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.68, 0.95, 0.99],
            plot_density = False,
            fill_contours = False,
            plot_datapoints = False
        )

        fig_final = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.68, 0.95, 0.99],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            fig = figure
        )
        
        plt.savefig(f"corner_plot_zs2_{i:.2f}.png")
        plt.close()    


def double_variable(a, b, step1, c, d, step2, lens):
    z_l = lens
    z_s1 = np.arange(a, b, step1)
    z_s2 = np.arange(c, d, step2)

    name = NUTS(model)
    mcmc = MCMC(name, num_warmup=1000, num_samples=10000, num_chains=1)
    key = jax.random.PRNGKey(100)


    w_0_desi = read_txt("chain.1.txt", 3)
    w_a_desi = read_txt("chain.1.txt", 4)
    OmM_desi = read_txt("chain.1.txt", 6)


    for i in z_s1:
        for j in z_s2:
            if j <= z_l or i <= z_l:
                pass
            elif j <= i:
                pass
            else:
                mcmc.run(key, z_l, i, j)
                #mcmc.print_summary()
            

                w_0_samples = mcmc.get_samples()['w_0']
                w_a_samples = mcmc.get_samples()['w_a']
                OmM_samples = mcmc.get_samples()['OmM']

                figure = corner.corner(
                    np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
                    labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
                    color = "red",
                    smooth = 0.5,
                    plot_contours=True,
                    levels=[0.68, 0.95, 0.99],
                    plot_density = False,
                    fill_contours = False,
                    plot_datapoints = False
                )

                fig_final = corner.corner(
                    np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
                    labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
                    color = "blue",
                    smooth = 0.5,
                    plot_contours=True,
                    levels=[0.68, 0.95, 0.99],
                    plot_density = False,
                    plot_datapoints = False,
                    fill_contours = True,
                    show_titles=True,
                    fig = figure
                )
                plt.savefig(f"corner_plot_zs1_{i:.2f}_zs2_{j:.2f}.png")
                plt.close()
    

def bare_bone():
    z_l = 0.222
    z_s1 = 0.609
    z_s2 = 2.035

    w_0_desi = read_txt("chain.1.txt", 3)
    w_a_desi = read_txt("chain.1.txt", 4)
    OmM_desi = read_txt("chain.1.txt", 6)

    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=1000, num_samples=1000000, num_chains=1)
    key = jax.random.PRNGKey(100)

    mcmc.run(key, z_l, z_s1, z_s2)

    w_0_samples = mcmc.get_samples()['w_0']
    w_a_samples = mcmc.get_samples()['w_a']
    OmM_samples = mcmc.get_samples()['OmM']
    desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
    desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
    desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values


    #joint likelihood
    joint = corner.corner(
        np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
        labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
        color = "black",
        #smooth = 0.5,
        plot_contours=True,
        levels=[0.68, 0.95, 0.99],
        plot_density = False,
        plot_datapoints = False,
        fill_contours = True,
        show_titles=True,
        weights=desi_likelihood
    )

    desi = corner.corner(
        np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
        color = "grey",
        smooth = 0.5,
        plot_contours=True,
        levels=[0.68, 0.95, 0.99],
        plot_density = False,
        fill_contours = False,
        plot_datapoints = False,
        fig = joint
    )

    fig_final = corner.corner(
        np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
        labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
        color = "blue",
        smooth = 0.5,
        plot_contours=True,
        levels=[0.68, 0.95, 0.99],
        plot_density = False,
        plot_datapoints = False,
        fill_contours = True,
        show_titles=True,
        fig = desi
    )

    plt.show()
    plt.close()



def vs1(start:float, stop:float, step:float, lens:float = 0.222, s2:float = 2.035, sample = 10000):
    #fixed values throughout this model, VARY THIS BEFORE RUNNING
    z_l = lens
    z_s2 = s2

    z_s1_range = np.arange(start, stop, step)

    if start <= z_l or stop >= z_s2:
        print("read idiot")
        return

    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=sample, num_chains=1)
    key = jax.random.PRNGKey(100)

    for z_s1 in z_s1_range:

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()
    
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        model = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            truths=[0.3, -1.0, 0.0]
        )

        desi = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "grey",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            fill_contours = False,
            plot_datapoints = False,
            fig = model
        )

        joint = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "black",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            weights=desi_likelihood,
            fig = desi
        )

        for i in model.axes:
            i.set_facecolor('none')


        plt.suptitle(f'z_l={z_l:.2f}  z_s1={z_s1:.2f}  z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"varying_s1={z_s1:.2f}.png")
        plt.close()

    return ("varying_s1=", start, stop, step)


def vs2(start:float, end:float, step:float, lens:float = 0.222, s1:float = 0.609, sample = 10000):
    #fixed values throughout this model
    z_l = lens
    z_s1 = s1

    z_s2_range = np.arange(start, end, step)

    if start <= z_s1:
        print("read idiot")
        return

    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=sample, num_chains=1)
    key = jax.random.PRNGKey(100)

    w_a_sigma = [] # store the sigma of w_a for each s2 value to plot later
    w_0_sigma = []
    beta_array = [] # store the beta values for each s2 value to plot later
    

    for z_s2 in z_s2_range:

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()

        
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        w_a_weighted_mean = np.average(w_a_samples, weights=desi_likelihood)
        w_a_weighted_variance = np.average((w_a_samples - w_a_weighted_mean)**2, weights=desi_likelihood)
        w_a_weighted_std = np.sqrt(w_a_weighted_variance)
        w_a_sigma.append(w_a_weighted_std/w_a_desi_std)


        w_0_weighted_mean = np.average(w_0_samples, weights=desi_likelihood)
        w_0_weighted_variance = np.average((w_0_samples - w_0_weighted_mean)**2, weights=desi_likelihood)
        w_0_weighted_std = np.sqrt(w_0_weighted_variance)
        w_0_sigma.append(w_0_weighted_std/w_0_desi_std)
        
        beta_array.append(mcmc.get_samples()['b'].mean())


        model = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            truths=[0.3, -1.0, 0.0],
            range=axis_range
        )

        desi = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "grey",
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            fill_contours = True,
            plot_datapoints = False,
            range=axis_range,
            fig = model
        )

        joint = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "black",
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            weights=desi_likelihood,
            range=axis_range,
            fig = desi
        )

        plt.suptitle(f'z_l={z_l:.2f}  z_s1={z_s1:.2f}  z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"corner_s2={z_s2:.2f}.png")
        plt.close()


        plt.hist(w_a_samples, bins=30, density=True, alpha=0.5, color='black', label='w_a samples', weights = desi_likelihood)
        plt.suptitle(f'w_a distribution for z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"wa_s2={z_s2:.3f}.png")
        plt.close()

    plt.plot(beta_array, w_a_sigma)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma_{w_a,DESI+DSPL}/\sigma_{DESI}$")
    plt.savefig(f"beta_vs_sigma,zs2={start}-{end}.png")

    plt.plot(beta_array, w_0_sigma)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma_{w_0,DESI+DSPL}/\sigma_{DESI}$")
    plt.savefig(f"beta_vs_sigmaw0,zs2={start}-{end}.png")

    print(beta_array)
    print(w_a_sigma)
    print(w_0_sigma)
    return ("varying_s2=", start, end, step)


def vl_scale(start:float, end:float, step:float, s1:float = 1.2, s2:float = 1.8, sample = 10000):


    z_l_range = np.arange(start, end, step)


    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=sample, num_chains=1)
    key = jax.random.PRNGKey(100)

    w_a_sigma = [] # store the sigma of w_a for each s2 value to plot later
    w_0_sigma = []
    beta_array = [] # store the beta values for each s2 value to plot later

    zl_array = []
    zs1_array = []
    zs2_array = []

    for z_l in z_l_range:

        z_s1 = z_l * s1
        z_s2 = z_l * s2

        zl_array.append(z_l)
        zs1_array.append(z_s1)
        zs2_array.append(z_s2)

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()
    
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        w_a_weighted_mean = np.average(w_a_samples, weights=desi_likelihood)
        w_a_weighted_variance = np.average((w_a_samples - w_a_weighted_mean)**2, weights=desi_likelihood)
        w_a_weighted_std = np.sqrt(w_a_weighted_variance)
        w_a_sigma.append(w_a_weighted_std/w_a_desi_std)

        w_0_weighted_mean = np.average(w_0_samples, weights=desi_likelihood)
        w_0_weighted_variance = np.average((w_0_samples - w_0_weighted_mean)**2, weights=desi_likelihood)
        w_0_weighted_std = np.sqrt(w_0_weighted_variance)
        w_0_sigma.append(w_0_weighted_std/w_0_desi_std)

        b_i = mcmc.get_samples()['b'].mean()
        
        beta_array.append(b_i)

        model = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            range=axis_range,
            truths=[0.3, -1.0, 0.0]
        )

        desi = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "yellow",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            fill_contours = False,
            plot_datapoints = False,
            range=axis_range,
            fig = model
        )

        joint = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "black",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            weights=desi_likelihood,
            range=axis_range,
            fig = desi
        )

        plt.suptitle(f'z_l={z_l:.2f}  z_s1={z_s1:.2f}  z_s2={z_s2:.2f} beta={b_i:.4f}', y=1, fontsize=14)
        plt.savefig(f"corner_scale_l={z_l:.2f}.png")
        plt.close()


        plt.hist(w_a_samples, bins=30, density=True, alpha=0.5, color='black', label='w_a samples', weights = desi_likelihood)
        plt.suptitle(f'w_a distribution for z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"wa_s2={z_s2:.2f}.png")
        plt.close()

    plt.plot(beta_array, w_a_sigma)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma_{w_a,DESI+DSPL}/\sigma_{DESI}$")
    plt.savefig(f"beta_vs_sigma,{start}-{end}.png")
    plt.close()

    plt.plot(zl_array, w_a_sigma, label = "lens")
    plt.plot(zs1_array, w_a_sigma, label = "source 1")
    plt.plot(zs2_array, w_a_sigma, label = "source 2")
    plt.legend()
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\sigma_{w_a,DESI+DSPL}/\sigma_{DESI}$")
    plt.savefig(f"z_vs_sigma,{start}-{end}.png")
    plt.close()

    print(beta_array)
    print(w_a_sigma)
    print(w_0_sigma)
    return ("scaling_l=", start, end, step)


def vl_offset(start:float, end:float, step:float, s1:float = 1, s2:float = 2, sample = 10000):


    z_l_range = np.arange(start, end, step)


    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=sample, num_chains=1)
    key = jax.random.PRNGKey(100)
    w_a_sigma = []
    beta_array = []

    for z_l in z_l_range:
        z_s1 = z_l + s1
        z_s2 = z_l + s2

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()

        
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        w_a_weighted_mean = np.average(w_a_samples, weights=desi_likelihood)
        w_a_weighted_variance = np.average((w_a_samples - w_a_weighted_mean)**2, weights=desi_likelihood)
        w_a_weighted_std = np.sqrt(w_a_weighted_variance)
        w_a_sigma.append(w_a_weighted_std/w_a_desi_std)
        
        beta_array.append(mcmc.get_samples()['b'].mean())


        model = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "blue",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            truths=[0.3, -1.0, 0.0]
        )

        desi = corner.corner(
            np.array([OmM_desi, w_0_desi, w_a_desi]).T, 
            color = "Yellow",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            fill_contours = False,
            plot_datapoints = False,
            fig = model
        )

        joint = corner.corner(
            np.array([OmM_samples, w_0_samples, w_a_samples]).T, 
            labels=[r"$\Omega_M$", r"$w_0$", r"$w_a$"], 
            color = "black",
            smooth = 0.5,
            plot_contours=True,
            levels=[0.3935, 0.8647, 0.9889],
            plot_density = False,
            plot_datapoints = False,
            fill_contours = True,
            show_titles=True,
            weights=desi_likelihood,
            fig = desi
        )

        plt.suptitle(f'z_l={z_l:.2f}  z_s1={z_s1:.2f}  z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"varying_s2={z_s2:.2f}.png")
        plt.close()


        plt.hist(w_a_samples, bins=30, density=True, alpha=0.5, color='black', label='w_a samples', weights = desi_likelihood)
        plt.suptitle(f'w_a distribution for z_s2={z_s2:.2f}', y=1, fontsize=14)
        plt.savefig(f"w_a_hist_s2={z_s2:.2f}.png")
        plt.close()

    plt.plot(beta_array, w_a_sigma)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sigma_{w_a,DESI+DSPL}/\sigma_{DESI}$")
    plt.show()
    return ("offset_l=", start, end, step)



def the_plot_that_dan_says_is_good(num:int, sample = 10000):

    name = NUTS(joint_model)
    mcmc = MCMC(name, num_warmup=3000, num_samples=sample, num_chains=1)
    key = jax.random.PRNGKey(100)

    w_a_sigma = [] # store the sigma of w_a for each s2 value to plot later
    w_0_sigma = []
    beta_array = [] # store the beta values for each s2 value to plot later

    zl_array = []
    zs1_array = []
    zs2_array = []
    z_mean = []


    for i in range(num):
        #z_l = npr.sample("z_l", dist.Uniform(0.2, 2))
        #z_s1 = npr.sample("z_s1", dist.Uniform(z_l, 4))
        #z_s2 = npr.sample("z_s2", dist.Uniform(z_s1, 6))

        z_l = np.random.uniform(0.2, 2)
        z_s1 = np.random.uniform(z_l, 4)
        z_s2 = np.random.uniform(z_s1, 6)

        m = np.mean([z_l, z_s1, z_s2])

        zl_array.append(z_l)
        zs1_array.append(z_s1)
        zs2_array.append(z_s2)
        z_mean.append(m)

        mcmc.run(key, z_l, z_s1, z_s2)
        #mcmc.print_summary()
    
        w_0_samples = mcmc.get_samples()['w_0']
        w_a_samples = mcmc.get_samples()['w_a']
        OmM_samples = mcmc.get_samples()['OmM']
        desi_likelihood = mcmc.get_samples()['desi_likelihood'].flatten()
        desi_likelihood = jnp.nan_to_num(desi_likelihood, neginf=-1e10)
        desi_likelihood = jnp.exp(desi_likelihood - jnp.max(desi_likelihood)) # normalize the likelihood values

        w_a_weighted_mean = np.average(w_a_samples, weights=desi_likelihood)
        w_a_weighted_variance = np.average((w_a_samples - w_a_weighted_mean)**2, weights=desi_likelihood)
        w_a_weighted_std = np.sqrt(w_a_weighted_variance)
        w_a_sigma.append(w_a_weighted_std/w_a_desi_std)

        w_0_weighted_mean = np.average(w_0_samples, weights=desi_likelihood)
        w_0_weighted_variance = np.average((w_0_samples - w_0_weighted_mean)**2, weights=desi_likelihood)
        w_0_weighted_std = np.sqrt(w_0_weighted_variance)
        w_0_sigma.append(w_0_weighted_std/w_0_desi_std)

        b_i = mcmc.get_samples()['b'].mean()
        beta_array.append(b_i)
    
    idx = np.argsort(z_mean)
    zl_sort, zs1_sort, zs2_sort = np.array(zl_array)[idx], np.array(zs1_array)[idx], np.array(zs2_array)[idx]


    i = np.linspace(0, num, num)
    plt.scatter(zl_sort, i, c=w_a_sigma, marker="o", label = "lens", cmap="berlin", vmin = 0, vmax = 2)
    plt.scatter(zs1_sort, i, c=w_a_sigma, marker="s", label = "source 1", cmap="berlin", vmin = 0, vmax = 2)
    plt.scatter(zs2_sort, i, c=w_a_sigma, marker="^", label = "source 2", cmap="berlin", vmin = 0, vmax = 2)
    plt.colorbar()
    plt.legend()
    plt.show()
    




def main():
    #insert model:
    #model = vs2(2.035, 2.036, 0.01, lens=0.222, s1=0.609,sample=300000)
    #model = vl_scale(0.2, 2.1, 0.2, s1=1.6, s2=7, sample = 100000)
    the_plot_that_dan_says_is_good(30)

    command = ""
'''
    try:
        while command not in ["Y", "N"]:
            command = input("M'lord, your humble servant has completed what you asked for, must I kill my own creations? (Y/N): ")
            command = command.upper()
    
        if command.upper() == "Y":
            for i in np.arange(model[1], model[2], model[3]):
                print(f"{model[0]}{i:.2f}.png")
                file_path = Path(f"{model[0]}{i:.2f}.png")
                file_path.unlink()

            print("Files deleted.")
    except KeyboardInterrupt:
        try:
            for i in np.arange(model[1], model[2], model[3]):
                    file_path = Path(f"{model[0]}{i:.2f}.png")
                    file_path.unlink()
        except FileNotFoundError:
            pass
        print("Process interrupted. Files deleted.")
'''

main()