# Singular Value Decomposition and the Empirical Spectral Distribution
## SVD visualisation

> In this section, we will be working in `./svd_visualization.py` and `./coords_and_svd.py`. We are also using zero-based indexing, because I am a programmer.

SVD is a tool which allows you to find which dimensions in a dataset have a lot of variance. What does that actually mean? Let's take the following set of 200 points in 3D space;

![The bare, unanalysed space](./gifs/bare-space.gif)

Our data initially exists without any obvious coordinate system applied to it, and it's spinning through space. Or perhaps we're spinning around it; the two are identical from our perspective. The goal is impose a coordinate system onto this data in which the zeroth coordinate has the most variance, the next coordinate has the next most, et cetera. We can see that in our case, the data is pancake shaped, with more variance in one direction than others. Let's take that direction as our zeroth 'basis vector', that is the zeroth coordinate in the system that we are constructing.

![The zeroth basis vector has been applied](./gifs/zeroth-basis-applied.gif)

What does this choice of basis vector mean in practise? It is chosen so that, if we were to pick one of our points in the data, simply knowing the closest point on this 1D line would give the best guess on where that point lies in the wider space. If we were to draw this line in a different direction, and there are infinitely many to choose from, we would have a worse guess. 

But even our mathematically optimal zeroth basis vector still can't yield a great guess, so how do we go about selecting another? Here, we can take advantage of a simple fact, our next basis vector has to be perpendicular to the zeroth. If it wasn't, there would be redundant information between the two basis vectors. This also arises from the fact that SVD computes a rotation, but that's a rabbit hole we can leave until later.

In 3D space, we still have an entire plane of vectors to choose from for our second basis vector.

![We have a plane of possibilities for the first basis vector](./gifs/first-basis-possibilities.gif)

We can, however, repeat a version of the same trick we applied to find the zeroth basis vector. We find the line which, knowing where the point sits on the plane defined by the zeroth and first basis vectors, tells us the most about where the point is in 3D space.

![The first basis vector has been applied](./gifs/first-basis-applied.gif)

Where do we go from here? Well, we are lucky enough to be living in a 3D space. In 3D space, given two vectors, there is only one vector perpendicular to both of them. Well, technically there are two (up and down) but they would simply yield an inverted sign. Let's apply that last basis vector now.

![The second basis vector has been applied](./gifs/second-basis-applied.gif)

Because we have a 3D input space, and a 3D output space, knowing the closest point on all of these three basis vectors tells us where the data is in 3D space. SVD is sometimes used for a type of lossy compression known as dimensional reduction. If you were to flatten all the points so that you only knew their position relative to the zeroth and first basis vector, you could still have a pretty good guess of where the point is in 3D space while only having to communicate two values. 

You can also now see how SVD computes a rotation, essentially constructing a complete coordinate system for our data. 

I have given myself some artistic license in this visualisation; the basis vectors are of different lengths. The basis vectors produced by SVD are always unit (1) length because SVD creates a rotation matrix without any scaling. To produce this visualisation, I have scaled each basis by a set of another set of outputs of the SVD algorithm, the `S` values. These are technically known as the singular values, but you can think of them as 'significance'. You can see that the zeroth is the highest significance, the first the next highest, and the final the least. If we had 100-Dimensional data, it would produce 100 perpendicular basis vectors with 100 significance levels, starting with the most and ending with the least significant.

> SVD is typically computed all at once rather than basis by basis, but this model is very useful in thinking about what exactly SVD is doing.

## ESD visualisation

> In this section, we will be working in `./esd_visualisation.py` and `./coords_and_svd.py`.

Vector spaces become very hard to visualise past 3 dimensions, but SVD remains useful. Let's focus on one type of analysis which can be performed, taking the Empirical Spectral Distribution (ESD).

When we use numpy to compute the SVD in `./coords_and_svd.py`, we get three cryptically named outputs;

```
U, S, Vh = np.linalg.svd(coords_world)
```

In reverse order:
 - `Vh` is the rotation matrix which maps whatever input space we used to the set of basis vectors we constructed above, where they are ranked from most to least variance. For the same set of input vectors, we can apply any rotation we want, and SVD is generally guaranteed to converge the same output space. This is rotation matrix is then applied in the next line;

```
coords_world = np.matmul(coords_world, Vh)
```

- `S` contains the singular values (I call them 'significance' above). It's how long we portrayed our basis vectors to be in the last visualisation. For each of the basis vectors in `Vh`, it tells us how important they are to describing the variance of the data.
- Finally, `U` represents, for each of our input data points, what combination of our constructed basis vectors can reconstruct them. Unlike `Vh` and `S`, this part of the analysis is not aggregated and gives one output per input vector.

ESD analysis only considers the values of `S`, which is a flat 1-dimensional vector associating significance levels to each of the basis vectors. Let's take those scaled vectors and lay them flat, horizontal, and starting from the same X coordinate.

![vectors laid flat](./imgs/esd-3d.png)

Now let's use that to build a histogram.

![vectors laid flat with histogram](./imgs/esd-3d-hist.png)

Et voila, we can see the worlds most pointless histogram. Because we only have three basis vectors to analyse, we have more buckets than we have values to fill them. Something to note here - it's a **density** histogram, where the area of each bucket corresponds to the number of items in the bucket rather than their height - this is the convention when creating an ESD histogram. 

Let's do something more interesting which shows the power of ESD to analyse very large vector spaces. Instead of working in a 3D space, let's instead work in a 1000D space, and then take 1000 vectors to analyse. The values in each row and column of this input matrix are all sampled from the same normal distribution. This means that they are statistically identical. They are also uncorrelated, meaning they are independent. This is a special type of matrix known as an independent and identically distributed (IID) matrix. When we look at distribution of the SVD singular values, we see;

![large dimensional histogram](./imgs/esd-norm-hist.png)

And here is the magic of the ESD; we can take a vector space far too large to be reasoned about visually, but still make meaningful conclusions about the structure of the underlying space. There is clearly a pattern here, and we can treat the singular values as a distribution and use all of the tools of statistics to analyse what's going on.

ESD analysis is usually introduced towards the beginning of a course on Random Matrix Theory (RMT), surrounded by derivations and notation. While those are vital, there also is a real risk of leaving such a course without any intuitive sense of what's actually going on inside of these operations and spaces. In particular, RMT has led to theories describing the internal structure of a neural network with tools like [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher). Neural networks are composed mostly of fancy matrix multiplications after all, and ESD analysis has shown an unreasonable level of generality in understanding how they work. It can do this without looking at training data or even knowing their intended purpose. Incidentally, my cookies have not recovered since discovering WeightWatcher.

Unfortunately, I have shown you the most boring type of ESD that exists. We've used a mathematical tool designed to measure correlation and structure to look at a matrix without any of either. The ESD distribution that we saw above in the IID normally distributed case follows the Marchenko–Pastur (MP) law. Let's look at a matrix which I prepared earlier.

![heavy tailed histogram](./imgs/esd-heavy-hist.png)

This is the mythical power law Heavy-Tailed matrix. Unfortunately, a lot of the detail has been blown out by the massive spike of low singular values. Let's fix that with a log scale; 

![heavy tailed histogram log](./imgs/esd-heavy-hist-log.png)

The distribution of singular values here follow a power law, which can be seen from all of those singular values hanging out on their own at the very high end. When a neural network is training well, your singular values tend to look a little like this. The specific type and shape of the power law can tell you a lot about how the layers of your networks are training.

> A confession; power law Heavy-Tailed matrices are confusing. This project (beautifully) taught me that I didn't understand them as well as I thought I did. The above section previously displayed a lognormal-ish matrix which failed to fit a power law at all. At one point, I even relented and constructed a correct one with gradient descent, which is machine learning speak for 'I don't know how to make this analytically'. The analytical approach I used in the end was actually very simple (albeit heavily commented), if you want to see it take a look at the block that starts `elif dist == 'heavy':` in `./esd_visualisation.py`.

Okay, I keep saying 'power law', but how do we know what that means? The power law distribution used here is parameterised by two variables - `x_min`, which dictates the lower bound, and `alpha`, which causes the tail to drop off faster when increased. If we use WeightWatcher's builtin fitter to measure the properties of the ESD's power law, we get the following output;

```
WWFit(power_law xmin: 10.0146, alpha: 2.6218, sigma: 0.0513, data: 999)
```

And what were the params of the power law we constructed the curve with?

```
x_min = 10
alpha = 2.5
```

That's not far off, but how close does this look in practise? Let's plot this fit line against the ground truth that we were drew our singular values from.

![heavy tailed histogram log and fitted power law](./imgs/esd-heavy-hist-log-pl.png)

Obviously, 'not far off' is not a statistically valid concept and we would need to test a lot of different `x_mins` and `alphas` to build more confidence than this simple gut check. From this example, however, we can see one of the things that makes power laws hard to fit;

 - At low values, there is a lot of data and the curve matches closely
 - At high values, the distribution thins out a lot, leading to larger and larger divergence between the fit curve and ground truth

This explains why `x_min` has been fitted much better than `alpha`.

Okay. Now let's try this analysis on a layer from a real, trained network; we're going to take the final layer from a pretrained AlexNet image classifier. This layer is dense, meaning it's implemented as a matrix multiplication by the layer weights followed by addition by the layer bias. Let's plot a histogram of the singular values of our weights without fitting a power law.

![ESD of the final layer of AlexNet](./imgs/esd-alexnet-hist-log.png)

We only have one more transformation to apply before we can fit our power law. ESDs come in a few flavours which encode the same information in different ways, we have SVD singular values (what we've been using so far) and the correlation matrix eigenvalues. The relationship between SVD singular values and correlation matrix eigenvalues are analogous to the relationship between standard deviation and variance. They encode the same information, but some statistical operations are expressed more naturally in one than the other. 

Given a sample of the SVD singular values, we can get a sample of the correlation matrix eigenvalues **simply by squaring them**. Another analogy to consider is the difference between speed and kinetic energy - some calculations are easier in one form than the other. Let's square our singular values, and fit a power law.

![ESD of the  final layer of AlexNet with a fitted power law](./imgs/esd-alexnet-hist-log-pl-sq.png)

Interesting. Something to notice is that when we're dealing with a real network, we rarely actually fit _all_ of the singular values. The is section below the fit `x_min` does not fit our power law. This is usually dismissed as noise in this analysis. Let's look at the parameters of our fit.

```
WWFit(power_law xmin: 2.0464, alpha: 3.0199, sigma: 0.1523, data: 176)
```

For more information on interpreting these alpha values, take a look at the references below.

### Further Reading:

 - [Traditional and Heavy-Tailed Self Regularization in Neural Network Models](https://arxiv.org/abs/1901.08276) (Martin & Mahoney, 2019)
 - [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) on GitHub
 - [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Krizhevsky, Sutskever & Hinton, 2012)
