Scripts in this folder:

a) computeEllipseParameters.py gives four output files (two for each lepton category - ElEl and MuMu):

    1) sigmas_MuMu(ElEl).json: contains, in order: mbb, mllbb, width_bb, and width_llbb, where mbb(llbb) is the mean of a 1D gaussian fit performed on the invariant mass distribution of bb(llbb) and width_bb(llbb) is the sigma (not resolution!) of the mentioned gaussian fit.

    2) ellipseParam_MuMu(ElEl).json: contains, in order: mbb, mllbb, a, b, theta, MA, MH, where mbb and mllbb are the recontructed masses described at point 1), a, b, and theta are the ellipses parameters coming from a 2D gaussian fit of the 2D mbb VS mllbb histograms, MA and MH are the simulated masses.
    
    Additional : if the option --fit is provided, the script will look at the pol2 fit parameters produce by the extrapolation (see point b) to fix the centroid 
    Additional : if the option --window is provided, the script will only take into accounts the points inside a window for the 2D fit (the diagonalization of the covariance matrix). The window corresponds to 50% of the sigma in the two directions for mlljj and mjj (hence rectangular), this value can be changed in the script (will also plot the effect of the window on the 2D distribution)

b) centroidExtrapolation.py makes the extrapolation of the points in the plane mA-m_bb and mH-m_llbb by only considering "physical" points (aka the ones that are not decreasing, meanign that the gaussian fit has failed). It also produces the surface plots of a (major axis), b minor axis and theta (tilt angle) not taking into account the removed points. The algorithm for removing the points consists in : for a given x, only take the couples (x,y) sich that y > average y of the previous x value 

c) plotDistributions.py produces two kinds of plots :
    - a plot for each mass point of the m_lljj and m_jj distributions (with or without the centroid fit, see below). It also adds the Gaussian fit from computeEllipseParameters.py
    - a 2D plot of the distribution (mjj,mlljj) and the ellipses of the different configurations (see below) for rho = 1,2,3
                    -> if option --fit is provided, the script will take the centroids from the fit (see point b) : use them for the 1D plots via the txt files and add the ellipse via the json files to the 2D plots (in addition to basic case)
                    -> if option --window is provided, the script will look at the json files corresponding to the window case (see point a) (only affects 2D plots, not the 1D) and add the ellipse to the others
                    -> if option --count is used, the script will take the ratio of the points inside and outside the ellipse from the json file (see point e) and add them to the legend


d) widthExtrapolation.py makes the extrapolation of the width as a function of the centroid (deprecated, might not be useful after all...)

e) getFractionInsideEllipse.py takes the root files containing the histograms (from launchfractory.py), ie the specific ones where is computed the number in and out of the ellipses, adds the ratio inside json files

f) tdrstyle.py is used as a template for the plotting style

