# Singular Value Decomposition visualisation

A tool to help wrap your head around what exactly the SVD does by explaining it as rotation, in this case in 3D space;

 - Our data initially exists without any obvious coordinate system applied to it, and it's spinning freely through space. We can see that the data is pancake shaped, with a lot more variance in one axis than the other two.
![Stage one](./gifs/s1.gif)
 - When the first boolean flag (`add_basis`) is flipped to true, we see a set of three basis vectors which would represent our coordinate system. but our data is still unconstrained by it.
 - When the second boolean flag (`constrain_first_axis`) is flipped to true, we see that the direction of maximum variance has been constrained to the first (red) basis vector, but it is still free to rotate around the other two.
 - When the third boolean flag (`constrain_second_axis`) is flipped to true, we see that all rotation has been removed. We have set the second basis vector to the one perpundicular to the first basis vector which describes the highest remaining variance.
 - The Third basis vector, therefore, takes up any variance not explained by the first two as we have no more dimensions left to play with.
- When the fourth boolean flag is flipped (`scale_basis_by_significance`), we see how much variance is explained by each basis.

Although SVD is not typically computed basis-by-basis, this mental model scales to higher dimensions and I found it very useful in understanding what exactly is going on.
