import numpy as np
import matplotlib.pyplot as plt
from coords_and_svd import n, coords_world, S, Vh
from weightwatcher.WW_powerlaw import WWFit

plot_hist = True
large_dim_dataset = True
dist = 'heavy'
fit_power_law = True
log = True
imshow = False

if large_dim_dataset:
    if dist == 'norm':
        coords_world = np.random.normal(size=[1000, 1000])
    elif dist == 'heavy':
        # This was an absolute can of wormss
        # The approach I ended up using:
        #  - Generate a heavy-tailed distribution to create target singular values
        #  - Create a matrix with those target singular values
        # More here:
        # https://blogs.sas.com/content/iml/2012/03/30/geneate-a-random-matrix-with-specified-eigenvalues.html
        
        # The powerlaw distribution is a funny one. It comes in two flavours;
        # - Bounded [x_min, x_max]
        # - Bounded [x_min, inf] <--- Used here.
        # It CANNOT be unbounded in both directions because the inverse CDF diverges at x=0.
        # It also permits an alpha value, increasing this makes the distribution fall off faster into the tail.
        # More here:
        # https://stats.stackexchange.com/a/406705
        
        # Generate target singular values:
        alpha = 2.5
        x_min = 10
        inv_cdf = lambda x: x_min * (x ** (-1 / (alpha - 1)))
        singular_values = inv_cdf(np.random.uniform(size=[1000]))
       
        # Generate matrix with target singular values
        sigma = np.diag(singular_values)
        # https://nhigham.com/2020/04/22/what-is-a-random-orthogonal-matrix/
        rand_ortho = lambda: np.linalg.qr(np.random.normal(size=[1000,1000]))[0]
        # rand_ortho = lambda: np.random.normal(size=[1000,1000]) # interestingly, this seems to create a *roughly* accurate alpha value (even if it creates a wildly off x_min)
        U = rand_ortho()
        V = rand_ortho()

        coords_world = np.matmul(U, np.matmul(sigma, np.transpose(V)))
    elif dist == 'alexnet':
        import torch
        import torchvision
        model = torch.hub.load('pytorch/vision:v0.24.1', 'alexnet', pretrained=True)
        model.eval() 
        coords_world = model.classifier[4].state_dict()['weight'].numpy()
        print(coords_world.shape)

    U, S, Vh = np.linalg.svd(coords_world)

if imshow:
    plt.imshow(coords_world)
    plt.colorbar()
    plt.show()
    exit()


if not large_dim_dataset:
    plt.plot([0,S[0]], [0.4,0.4], 'r')
    plt.plot([0,S[1]], [0.5,0.5], 'g')
    plt.plot([0,S[2]], [0.6,0.6], 'b')
    plt.ylim(0,1.5)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Singular value')

if plot_hist:
    plt.hist(S, bins=50, color='black', density=True, log=log)
    plt.ylabel('Count')
    if fit_power_law:
        fit = WWFit(S)
        print(fit)
        #print(dir(fit))
        plt.vlines(fit.xmin, *plt.ylim())

else:    
    ax.spines['left'].set_visible(False)
    plt.yticks([])

#plt.savefig('imgs/esd%s' % ('-hist' if hist else '')

#plt.savefig('imgs/esd-hist-%r-large-%r-dist-%s%s.png' % (plot_hist, large_dim_dataset, dist, '-log' if log else ''))
plt.show()
