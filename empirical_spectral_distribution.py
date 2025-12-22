import numpy as np
import matplotlib.pyplot as plt
from coords_and_svd import coords_world
from weightwatcher.WW_powerlaw import WWFit

dist = 'heavy'
plot_hist = True
log = True
fit_power_law = True

alpha = 2.5
x_min = 10

if dist == '3d':
    coords = coords_world
elif dist == 'norm':
    coords = np.random.normal(size=[1000, 1000])
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
    inv_cdf = lambda x: x_min * (x ** (-1 / (alpha - 1)))
    singular_values = inv_cdf(np.random.uniform(size=[1000]))
   
    # Generate matrix with target singular values
    sigma = np.diag(singular_values)
    # https://nhigham.com/2020/04/22/what-is-a-random-orthogonal-matrix/
    rand_ortho = lambda: np.linalg.qr(np.random.normal(size=[1000,1000]))[0]
    # rand_ortho = lambda: np.random.normal(size=[1000,1000]) # interestingly, this seems to create a *roughly* accurate alpha value (even if it creates a wildly off x_min)
    U = rand_ortho()
    V = rand_ortho()

    coords = np.matmul(U, np.matmul(sigma, np.transpose(V)))
elif dist == 'alexnet':
    import torch
    import torchvision
    model = torch.hub.load('pytorch/vision:v0.24.1', 'alexnet', pretrained=True)
    model.eval() 
    coords = model.classifier[4].state_dict()['weight'].numpy()
else:
    raise Exception('unrecognised distribution %s' % dist)

U, S, Vh = np.linalg.svd(coords)

if dist == '3d':
    plt.plot([0,S[0]], [0.013, 0.013], 'r')
    plt.plot([0,S[1]], [0.014, 0.014], 'g')
    plt.plot([0,S[2]], [0.015, 0.015], 'b')
    plt.ylim(0,0.018)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Singular value')

if plot_hist:
    plt.hist(S, bins=50, color='black', density=True, log=log)
    plt.ylabel('Density')
    if fit_power_law:
        fit = WWFit(S)
        print('Power law fit', fit)
        
        pdf_fit = lambda x: ((fit.alpha - 1) / fit.xmin) * (x / fit.xmin) ** (-fit.alpha)
        pdf_ground = lambda x: ((alpha - 1) / x_min) * (x / x_min) ** (-alpha)
        xs = np.linspace(fit.xmin, np.max(S), 100)

        plt.plot(xs, pdf_ground(xs), 'r--', label='Ground truth power law')
        plt.plot(xs, pdf_fit(xs), 'r', label='Fit power law')

        plt.vlines(fit.xmin, *plt.ylim(), 'blue', label='Fit x_min')
        plt.legend()
else:    
    ax.spines['left'].set_visible(False)
    plt.yticks([])

plt.savefig('imgs/esd%s%s%s%s.png' % (
    '-' + dist,
    '-hist' if plot_hist else '',
    '-log' if log else '',
    '-pl' if fit_power_law else ''
))

plt.show()
