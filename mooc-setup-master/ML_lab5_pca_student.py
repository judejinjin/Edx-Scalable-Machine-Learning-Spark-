
labVersion = 'cs190_week5_v_1_2'

import matplotlib.pyplot as plt
import numpy as np

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def create2DGaussian(mn, sigma, cov, n):
    """Randomly sample points from a two-dimensional Gaussian distribution"""
    np.random.seed(142)
    return np.random.multivariate_normal(np.array([mn, mn]), np.array([[sigma, cov], [cov, sigma]]), n)

dataRandom = create2DGaussian(mn=50, sigma=1, cov=0, n=100)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45, 54.5), ax.set_ylim(45, 54.5)
plt.scatter(dataRandom[:,0], dataRandom[:,1], s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
pass

dataCorrelated = create2DGaussian(mn=50, sigma=1, cov=.9, n=100)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
plt.scatter(dataCorrelated[:,0], dataCorrelated[:,1], s=14**2, c='#d6ebf2',
            edgecolors='#8cbfd0', alpha=0.75)
pass

# TODO: Replace <FILL IN> with appropriate code
correlatedData = sc.parallelize(dataCorrelated)
#print format(correlatedData)
meanCorrelated = correlatedData.mean()
print format(meanCorrelated)
correlatedDataZeroMean = correlatedData.map(lambda x: (np.subtract(x,meanCorrelated)))

print meanCorrelated
print correlatedData.take(1)
print correlatedDataZeroMean.take(1)

# TEST Interpreting PCA (1a)
from test_helper import Test
Test.assertTrue(np.allclose(meanCorrelated, [49.95739037, 49.97180477]),
                'incorrect value for meanCorrelated')
Test.assertTrue(np.allclose(correlatedDataZeroMean.take(1)[0], [-0.28561917, 0.10351492]),
                'incorrect value for correlatedDataZeroMean')

# TODO: Replace <FILL IN> with appropriate code
# Compute the covariance matrix using outer products and correlatedDataZeroMean
n= correlatedDataZeroMean.count()
#print format(correlatedDataZeroMean.map(lambda x: np.outer(x, x)).take(1))
correlatedCov = correlatedDataZeroMean.map(lambda x: np.outer(x, x)).sum() / n
print format(correlatedCov)

# TEST Sample covariance matrix (1b)
covResult = [[ 0.99558386,  0.90148989], [0.90148989, 1.08607497]]
Test.assertTrue(np.allclose(covResult, correlatedCov), 'incorrect value for correlatedCov')

# TODO: Replace <FILL IN> with appropriate code
def estimateCovariance(data):
    """Compute the covariance matrix for a given rdd.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input `RDD`.
    """
    meanArray = data.mean()

    dataCentered = data.map(lambda x: (np.subtract(x,meanArray)))

    n= data.count()
    #print format(correlatedDataZeroMean.map(lambda x: np.outer(x, x)).take(1))
    return dataCentered.map(lambda x: np.outer(x, x)).sum() / n

correlatedCovAuto= estimateCovariance(correlatedData)
print correlatedCovAuto

# TEST Covariance function (1c)
correctCov = [[ 0.99558386,  0.90148989], [0.90148989, 1.08607497]]
Test.assertTrue(np.allclose(correctCov, correlatedCovAuto),
                'incorrect value for correlatedCovAuto')

# TODO: Replace <FILL IN> with appropriate code
from numpy.linalg import eigh

# Calculate the eigenvalues and eigenvectors from correlatedCovAuto
eigVals, eigVecs = eigh(correlatedCovAuto)
print 'eigenvalues: {0}'.format(eigVals)
print '\neigenvectors: \n{0}'.format(eigVecs)

# Use np.argsort to find the top eigenvector based on the largest eigenvalue
inds = np.argsort(eigVals)
eigVecs = eigVecs[inds]
eigVecs = eigVecs[::-1]

topComponent = eigVecs[0]

eigVals = eigVals[inds]
eigVals = eigVals[::-1]
print 'eigenvalues: {0}'.format(eigVals)
print '\ntop principal component: {0}'.format(topComponent)

# TEST Eigendecomposition (1d)
def checkBasis(vectors, correct):
    return np.allclose(vectors, correct) or np.allclose(np.negative(vectors), correct)
Test.assertTrue(checkBasis(topComponent, [0.68915649, 0.72461254]),
                'incorrect value for topComponent')

# TODO: Replace <FILL IN> with appropriate code
# Use the topComponent and the data from correlatedData to generate PCA scores
correlatedDataScores = correlatedData.map(lambda x: np.dot(x, topComponent))
print 'one-dimensional data (first three):\n{0}'.format(np.asarray(correlatedDataScores.take(3)))

# TEST PCA Scores (1e)
firstThree = [70.51682806, 69.30622356, 71.13588168]
Test.assertTrue(checkBasis(correlatedDataScores.take(3), firstThree),
                'incorrect value for correlatedDataScores')

# TODO: Replace <FILL IN> with appropriate code
def pca(data, k=2):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
            scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals
            `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
            of length `k`.  Eigenvalues is an array of length d (the number of features).
    """
    meanArray = data.mean()

    dataCentered = data.map(lambda x: (np.subtract(x,meanArray)))

    n= data.count()
    #print "number of data: " + format(n)
    #print format(correlatedDataZeroMean.map(lambda x: np.outer(x, x)).take(1))
    covMatrix = dataCentered.map(lambda x: np.outer(x, x)).sum() / float(n)

    eigVals, eigVecs = eigh(covMatrix)
    inds = np.argsort(eigVals)

    #print format(eigVals)
    #print format(eigVecs)
    
    eigVals = eigVals[inds][::-1]
    eigVecs = eigVecs.T[inds][::-1]

    #print "after sorting:\n"
    #print format(eigVals)
    #print format(eigVecs)
    
    eigVals = eigVals.T
    eigVecs = eigVecs.T

    #print "after transposing:\n"
    #print format(eigVals)
    #print format(eigVecs)
    
    if k < len(eigVals):
        eigVals[k:len(eigVals)] = 0
        eigVecs[:, k:len(eigVals)] = 0
    
    #print format(eigVals[0:k])
    #print format(eigVecs[:, 0:k])
    #print format([[0.68915649,  0.72461254], [-0.72461254, 0.68915649]])
    dataScores = data.map(lambda x: np.dot(x, eigVecs)[0:k])
    
    return (eigVecs[:,0:k], dataScores,   eigVals[0:k])

# Run pca on correlatedData with k = 2
topComponentsCorrelated, correlatedDataScoresAuto, eigenvaluesCorrelated = pca(correlatedData)

# Note that the 1st principal component is in the first column
print 'topComponentsCorrelated: \n{0}'.format(topComponentsCorrelated.T)
print ('\ncorrelatedDataScoresAuto (first three): \n{0}'
       .format('\n'.join(map(str, correlatedDataScoresAuto.take(3)))))
print '\neigenvaluesCorrelated: \n{0}'.format(eigenvaluesCorrelated)

# Create a higher dimensional test set
pcaTestData = sc.parallelize([np.arange(x, x + 4) for x in np.arange(0, 20, 4)])
componentsTest, testScores, eigenvaluesTest = pca(pcaTestData, 3)

print '\npcaTestData: \n{0}'.format(np.array(pcaTestData.collect()))
print '\ncomponentsTest: \n{0}'.format(componentsTest)
print format(np.dot(componentsTest[0], componentsTest[0]))
print ('\ntestScores (first three): \n{0}'
       .format('\n'.join(map(str, testScores.take(3)))))
print '\neigenvaluesTest: \n{0}'.format(eigenvaluesTest)

print format(testScores.first()[0])
topComponentsCorrelatedK1, correlatedDataScoresK1, eigenvaluesCorrelatedK1 = pca(correlatedData, 1)

print format(topComponentsCorrelatedK1)
print format(eigenvaluesCorrelatedK1)
print format(correlatedDataScoresK1.take(3))


# TEST PCA Function (2a)
Test.assertTrue(checkBasis(topComponentsCorrelated.T,
                           [[0.68915649,  0.72461254], [-0.72461254, 0.68915649]]),
                'incorrect value for topComponentsCorrelated')
firstThreeCorrelated = [[70.51682806, 69.30622356, 71.13588168], [1.48305648, 1.5888655, 1.86710679]]
Test.assertTrue(np.allclose(firstThreeCorrelated,
                            np.vstack(np.abs(correlatedDataScoresAuto.take(3))).T),
                'incorrect value for firstThreeCorrelated')
Test.assertTrue(np.allclose(eigenvaluesCorrelated, [1.94345403, 0.13820481]),
                           'incorrect values for eigenvaluesCorrelated')
topComponentsCorrelatedK1, correlatedDataScoresK1, eigenvaluesCorrelatedK1 = pca(correlatedData, 1)
Test.assertTrue(checkBasis(topComponentsCorrelatedK1.T, [0.68915649,  0.72461254]),
               'incorrect value for components when k=1')
Test.assertTrue(np.allclose([70.51682806, 69.30622356, 71.13588168],
                            np.vstack(np.abs(correlatedDataScoresK1.take(3))).T),
                'incorrect value for scores when k=1')
Test.assertTrue(np.allclose(eigenvaluesCorrelatedK1, [1.94345403]),
                           'incorrect values for eigenvalues when k=1')
Test.assertTrue(checkBasis(componentsTest.T[0], [ .5, .5, .5, .5]),
                'incorrect value for componentsTest')
Test.assertTrue(np.allclose(np.abs(testScores.first()[0]), 3.),
                'incorrect value for testScores')
Test.assertTrue(np.allclose(eigenvaluesTest, [ 128, 0, 0 ]), 'incorrect value for eigenvaluesTest')

# TODO: Replace <FILL IN> with appropriate code
randomData = sc.parallelize(dataRandom)

# Use pca on randomData
topComponentsRandom, randomDataScoresAuto, eigenvaluesRandom = pca(randomData, 2)

print 'topComponentsRandom: \n{0}'.format(topComponentsRandom)
print ('\nrandomDataScoresAuto (first three): \n{0}'
       .format('\n'.join(map(str, randomDataScoresAuto.take(3)))))
print '\neigenvaluesRandom: \n{0}'.format(eigenvaluesRandom)

# TEST PCA on `dataRandom` (2b)
Test.assertTrue(checkBasis(topComponentsRandom.T,
                           [[-0.2522559 ,  0.96766056], [-0.96766056,  -0.2522559]]),
                'incorrect value for topComponentsRandom')
firstThreeRandom = [[36.61068572,  35.97314295,  35.59836628],
                    [61.3489929 ,  62.08813671,  60.61390415]]
Test.assertTrue(np.allclose(firstThreeRandom, np.vstack(np.abs(randomDataScoresAuto.take(3))).T),
                'incorrect value for randomDataScoresAuto')
Test.assertTrue(np.allclose(eigenvaluesRandom, [1.4204546, 0.99521397]),
                            'incorrect value for eigenvaluesRandom')

def projectPointsAndGetLines(data, components, xRange):
    """Project original data onto first component and get line details for top two components."""
    topComponent= components[:, 0]
    slope1, slope2 = components[1, :2] / components[0, :2]

    means = data.mean()[:2]
    demeaned = data.map(lambda v: v - means)
    projected = demeaned.map(lambda v: (v.dot(topComponent) /
                                        topComponent.dot(topComponent)) * topComponent)
    remeaned = projected.map(lambda v: v + means)
    x1,x2 = zip(*remeaned.collect())

    lineStartP1X1, lineStartP1X2 = means - np.asarray([xRange, xRange * slope1])
    lineEndP1X1, lineEndP1X2 = means + np.asarray([xRange, xRange * slope1])
    lineStartP2X1, lineStartP2X2 = means - np.asarray([xRange, xRange * slope2])
    lineEndP2X1, lineEndP2X2 = means + np.asarray([xRange, xRange * slope2])

    return ((x1, x2), ([lineStartP1X1, lineEndP1X1], [lineStartP1X2, lineEndP1X2]),
            ([lineStartP2X1, lineEndP2X1], [lineStartP2X2, lineEndP2X2]))

((x1, x2), (line1X1, line1X2), (line2X1, line2X2)) = \
    projectPointsAndGetLines(correlatedData, topComponentsCorrelated, 5)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2), figsize=(7, 7))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
plt.plot(line1X1, line1X2, linewidth=3.0, c='#8cbfd0', linestyle='--')
plt.plot(line2X1, line2X2, linewidth=3.0, c='#d6ebf2', linestyle='--')
plt.scatter(dataCorrelated[:,0], dataCorrelated[:,1], s=14**2, c='#d6ebf2',
            edgecolors='#8cbfd0', alpha=0.75)
plt.scatter(x1, x2, s=14**2, c='#62c162', alpha=.75)
pass

((x1, x2), (line1X1, line1X2), (line2X1, line2X2)) = \
    projectPointsAndGetLines(randomData, topComponentsRandom, 5)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2), figsize=(7, 7))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
plt.plot(line1X1, line1X2, linewidth=3.0, c='#8cbfd0', linestyle='--')
plt.plot(line2X1, line2X2, linewidth=3.0, c='#d6ebf2', linestyle='--')
plt.scatter(dataRandom[:,0], dataRandom[:,1], s=14**2, c='#d6ebf2',
            edgecolors='#8cbfd0', alpha=0.75)
plt.scatter(x1, x2, s=14**2, c='#62c162', alpha=.75)
pass

from mpl_toolkits.mplot3d import Axes3D

m = 100
mu = np.array([50, 50, 50])
r1_2 = 0.9
r1_3 = 0.7
r2_3 = 0.1
sigma1 = 5
sigma2 = 20
sigma3 = 20
c = np.array([[sigma1 ** 2, r1_2 * sigma1 * sigma2, r1_3 * sigma1 * sigma3],
             [r1_2 * sigma1 * sigma2, sigma2 ** 2, r2_3 * sigma2 * sigma3],
             [r1_3 * sigma1 * sigma3, r2_3 * sigma2 * sigma3, sigma3 ** 2]])
np.random.seed(142)
dataThreeD = np.random.multivariate_normal(mu, c, m)

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
norm = Normalize()
cmap = get_cmap("Blues")
clrs = cmap(np.array(norm(dataThreeD[:,2])))[:,0:3]

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(121, projection='3d')
ax.azim=-100
ax.scatter(dataThreeD[:,0], dataThreeD[:,1], dataThreeD[:,2], c=clrs, s=14**2)

xx, yy = np.meshgrid(np.arange(-15, 10, 1), np.arange(-50, 30, 1))
normal = np.array([0.96981815, -0.188338, -0.15485978])
z = (-normal[0] * xx - normal[1] * yy) * 1. / normal[2]
xx = xx + 50
yy = yy + 50
z = z + 50

ax.set_zlim((-20, 120)), ax.set_ylim((-20, 100)), ax.set_xlim((30, 75))
ax.plot_surface(xx, yy, z, alpha=.10)

ax = fig.add_subplot(122, projection='3d')
ax.azim=10
ax.elev=20
#ax.dist=8
ax.scatter(dataThreeD[:,0], dataThreeD[:,1], dataThreeD[:,2], c=clrs, s=14**2)

ax.set_zlim((-20, 120)), ax.set_ylim((-20, 100)), ax.set_xlim((30, 75))
ax.plot_surface(xx, yy, z, alpha=.1)
plt.tight_layout()
pass

# TODO: Replace <FILL IN> with appropriate code
threeDData = sc.parallelize(dataThreeD)
componentsThreeD, threeDScores, eigenvaluesThreeD = pca(threeDData,2)

print 'componentsThreeD: \n{0}'.format(componentsThreeD)
print ('\nthreeDScores (first three): \n{0}'
       .format('\n'.join(map(str, threeDScores.take(3)))))
print '\neigenvaluesThreeD: \n{0}'.format(eigenvaluesThreeD)
print format(np.sum(threeDScores.take(3)))
print format(np.sum(eigenvaluesThreeD))
print format(np.sum(componentsThreeD))

# TEST 3D to 2D (2c)
Test.assertEquals(componentsThreeD.shape, (3, 2), 'incorrect shape for componentsThreeD')
Test.assertTrue(np.allclose(np.sum(eigenvaluesThreeD), 969.796443367),
                'incorrect value for eigenvaluesThreeD')
Test.assertTrue(np.allclose(np.abs(np.sum(componentsThreeD)), 1.77238943258),
                'incorrect value for componentsThreeD')
Test.assertTrue(np.allclose(np.abs(np.sum(threeDScores.take(3))), 237.782834092),
                'incorrect value for threeDScores')

scoresThreeD = np.asarray(threeDScores.collect())

# generate layout and plot data
fig, ax = preparePlot(np.arange(20, 150, 20), np.arange(-40, 110, 20))
ax.set_xlabel(r'New $x_1$ values'), ax.set_ylabel(r'New $x_2$ values')
ax.set_xlim(5, 150), ax.set_ylim(-45, 50)
plt.scatter(scoresThreeD[:,0], scoresThreeD[:,1], s=14**2, c=clrs, edgecolors='#8cbfd0', alpha=0.75)
pass

# TODO: Replace <FILL IN> with appropriate code
def varianceExplained(data, k=1):
    """Calculate the fraction of variance explained by the top `k` eigenvectors.

    Args:
        data (RDD of np.ndarray): An RDD that contains NumPy arrays which store the
            features for an observation.
        k: The number of principal components to consider.

    Returns:
        float: A number between 0 and 1 representing the percentage of variance explained
            by the top `k` eigenvectors.
    """
    n = len(data.take(1)[0])
    components, scores, eigenvalues = pca(data, n)
    
    kOfArray = eigenvalues[0:k]
    allOfArray = eigenvalues
    
    return kOfArray.sum() / float(allOfArray.sum())

varianceRandom1 = varianceExplained(randomData, 1)
varianceCorrelated1 = varianceExplained(correlatedData, 1)
varianceRandom2 = varianceExplained(randomData, 2)
varianceCorrelated2 = varianceExplained(correlatedData, 2)
varianceThreeD2 = varianceExplained(threeDData, 2)
print ('Percentage of variance explained by the first component of randomData: {0:.1f}%'
       .format(varianceRandom1 * 100))
print ('Percentage of variance explained by both components of randomData: {0:.1f}%'
       .format(varianceRandom2 * 100))
print ('\nPercentage of variance explained by the first component of correlatedData: {0:.1f}%'.
       format(varianceCorrelated1 * 100))
print ('Percentage of variance explained by both components of correlatedData: {0:.1f}%'
       .format(varianceCorrelated2 * 100))
print ('\nPercentage of variance explained by the first two components of threeDData: {0:.1f}%'
       .format(varianceThreeD2 * 100))

# TEST Variance explained (2d)
Test.assertTrue(np.allclose(varianceRandom1, 0.588017172066), 'incorrect value for varianceRandom1')
Test.assertTrue(np.allclose(varianceCorrelated1, 0.933608329586),
                'incorrect value for varianceCorrelated1')
Test.assertTrue(np.allclose(varianceRandom2, 1.0), 'incorrect value for varianceRandom2')
Test.assertTrue(np.allclose(varianceCorrelated2, 1.0), 'incorrect value for varianceCorrelated2')
Test.assertTrue(np.allclose(varianceThreeD2, 0.993967356912), 'incorrect value for varianceThreeD2')

import os
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'neuro.txt')

inputFile = os.path.join(baseDir, inputPath)

lines = sc.textFile(inputFile)
print lines.first()[0:100]

# Check that everything loaded properly
assert len(lines.first()) == 1397
assert lines.count() == 46460

# TODO: Replace <FILL IN> with appropriate code
def parse(line):
    """Parse the raw data into a (`tuple`, `np.ndarray`) pair.

    Note:
        You should store the pixel coordinates as a tuple of two ints and the elements of the pixel intensity
        time series as an np.ndarray of floats.

    Args:
        line (str): A string representing an observation.  Elements are separated by spaces.  The
            first two elements represent the coordinates of the pixel, and the rest of the elements
            represent the pixel intensity over time.

    Returns:
        tuple of tuple, np.ndarray: A (coordinate, pixel intensity array) `tuple` where coordinate is
            a `tuple` containing two values and the pixel intensity is stored in an NumPy array
            which contains 240 values.
    """
    cells = line.split()
    coords = (int(float(cells[0])), int(float(cells[1])))
    return (coords, np.array(map(lambda x:float(x), cells[2:])))
    
rawData = lines.map(parse)
rawData.cache()
entry = rawData.first()
print 'Length of movie is {0} seconds'.format(len(entry[1]))
print 'Number of pixels in movie is {0:,}'.format(rawData.count())
print ('\nFirst entry of rawData (with only the first five values of the NumPy array):\n({0}, {1})'
       .format(entry[0], entry[1][:5]))

# TEST Parse the data (3b)
Test.assertTrue(isinstance(entry[0], tuple), "entry's key should be a tuple")
Test.assertEquals(len(entry), 2, 'entry should have a key and a value')
Test.assertTrue(isinstance(entry[0][1], int), 'coordinate tuple should contain ints')
Test.assertEquals(len(entry[0]), 2, "entry's key should have two values")
Test.assertTrue(isinstance(entry[1], np.ndarray), "entry's value should be an np.ndarray")
Test.assertTrue(isinstance(entry[1][0], np.float), 'the np.ndarray should consist of np.float values')
Test.assertEquals(entry[0], (0, 0), 'incorrect key for entry')
Test.assertEquals(entry[1].size, 240, 'incorrect length of entry array')
Test.assertTrue(np.allclose(np.sum(entry[1]), 24683.5), 'incorrect values in entry array')

# TODO: Replace <FILL IN> with appropriate code
mn = rawData.map(lambda point: max(point[1])).max()
mx = rawData.map(lambda point: min(point[1])).min()

print mn, mx

# TEST Min and max flouresence (3c)
Test.assertTrue(np.allclose(mn, 100.6), 'incorrect value for mn')
Test.assertTrue(np.allclose(mx, 940.8), 'incorrect value for mx')

example = rawData.filter(lambda (k, v): np.std(v) > 100).values().first()

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(300, 800, 100))
ax.set_xlabel(r'time'), ax.set_ylabel(r'flouresence')
ax.set_xlim(-20, 270), ax.set_ylim(270, 730)
plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
pass

# TODO: Replace <FILL IN> with appropriate code
def rescale(ts):
    """Take a np.ndarray and return the standardized array by subtracting and dividing by the mean.

    Note:
        You should first subtract the mean and then divide by the mean.

    Args:
        ts (np.ndarray): Time series data (`np.float`) representing pixel intensity.

    Returns:
        np.ndarray: The times series adjusted by subtracting the mean and dividing by the mean.
    """
    m = np.mean(ts)
    scaledTs = np.array(map(lambda x: (x-m)/m, ts))
    #print format(scaledTs)
    return scaledTs

#print format(rescale(rawData.take(1)[0]))
scaledData = rawData.mapValues(lambda v: rescale(v))
mnScaled = scaledData.map(lambda x: min(x[1])).min()
mxScaled = scaledData.map(lambda x: max(x[1])).max()
print mnScaled, mxScaled

# TEST Fractional signal change (3d)
Test.assertTrue(isinstance(scaledData.first()[1], np.ndarray), 'incorrect type returned by rescale')
Test.assertTrue(np.allclose(mnScaled, -0.27151288), 'incorrect value for mnScaled')
Test.assertTrue(np.allclose(mxScaled, 0.90544876), 'incorrect value for mxScaled')

example = scaledData.filter(lambda (k, v): np.std(v) > 0.1).values().first()

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(-.1, .6, .1))
ax.set_xlabel(r'time'), ax.set_ylabel(r'flouresence')
ax.set_xlim(-20, 260), ax.set_ylim(-.12, .52)
plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
pass

# TODO: Replace <FILL IN> with appropriate code
# Run pca using scaledData
componentsScaled, scaledScores, eigenvaluesScaled = pca(scaledData.map(lambda (k, v): v), 3)
print format(componentsScaled)
print format(scaledScores)
print format(eigenvaluesScaled)

# TEST PCA on the scaled data (3e)
print format(sum(eigenvaluesScaled))
Test.assertEquals(componentsScaled.shape, (240, 3), 'incorrect shape for componentsScaled')
Test.assertTrue(np.allclose(np.abs(np.sum(componentsScaled[:5, :])), 0.283150995232),
                'incorrect value for componentsScaled')
Test.assertTrue(np.allclose(np.abs(np.sum(scaledScores.take(3))), 0.0285507449251),
                'incorrect value for scaledScores')
Test.assertTrue(np.allclose(np.sum(eigenvaluesScaled[:5]), 0.206987501564),
                'incorrect value for eigenvaluesScaled')

import matplotlib.cm as cm

scoresScaled = np.vstack(scaledScores.collect())
imageOneScaled = scoresScaled[:,0].reshape(230, 202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Top Principal Component', color='#888888')
image = plt.imshow(imageOneScaled,interpolation='nearest', aspect='auto', cmap=cm.gray)
pass

imageTwoScaled = scoresScaled[:,1].reshape(230, 202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Second Principal Component', color='#888888')
image = plt.imshow(imageTwoScaled,interpolation='nearest', aspect='auto', cmap=cm.gray)
pass

# Adapted from python-thunder's Colorize.transform where cmap='polar'.
# Checkout the library at: https://github.com/thunder-project/thunder and
# http://thunder-project.org/

def polarTransform(scale, img):
    """Convert points from cartesian to polar coordinates and map to colors."""
    from matplotlib.colors import hsv_to_rgb

    img = np.asarray(img)
    dims = img.shape

    phi = ((np.arctan2(-img[0], -img[1]) + np.pi/2) % (np.pi*2)) / (2 * np.pi)
    rho = np.sqrt(img[0]**2 + img[1]**2)
    saturation = np.ones((dims[1], dims[2]))

    out = hsv_to_rgb(np.dstack((phi, saturation, scale * rho)))

    return np.clip(out * scale, 0, 1)

# Show the polar mapping from principal component coordinates to colors.
x1AbsMax = np.max(np.abs(imageOneScaled))
x2AbsMax = np.max(np.abs(imageTwoScaled))

numOfPixels = 300
x1Vals = np.arange(-x1AbsMax, x1AbsMax, (2 * x1AbsMax) / numOfPixels)
x2Vals = np.arange(x2AbsMax, -x2AbsMax, -(2 * x2AbsMax) / numOfPixels)
x2Vals.shape = (numOfPixels, 1)

x1Data = np.tile(x1Vals, (numOfPixels, 1))
x2Data = np.tile(x2Vals, (1, numOfPixels))

# Try changing the first parameter to lower values
polarMap = polarTransform(2.0, [x1Data, x2Data])

gridRange = np.arange(0, numOfPixels + 25, 25)
fig, ax = preparePlot(gridRange, gridRange, figsize=(9.0, 7.2), hideLabels=True)
image = plt.imshow(polarMap, interpolation='nearest', aspect='auto')
ax.set_xlabel('Principal component one'), ax.set_ylabel('Principal component two')
gridMarks = (2 * gridRange / float(numOfPixels) - 1.0)
x1Marks = x1AbsMax * gridMarks
x2Marks = -x2AbsMax * gridMarks
ax.get_xaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x1Marks))
ax.get_yaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x2Marks))
pass

# Use the same transformation on the image data
# Try changing the first parameter to lower values
brainmap = polarTransform(2, [imageOneScaled, imageTwoScaled])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap,interpolation='nearest', aspect='auto')
pass

# TODO: Replace <FILL IN> with appropriate code
vector = np.array([0., 1., 2., 3., 4., 5.])

# Create a multi-dimensional array that when multiplied (using .dot) against vector, results in
# a two element array where the first element is the sum of the 0, 2, and 4 indexed elements of
# vector and the second element is the sum of the 1, 3, and 5 indexed elements of vector.
# This should be a 2 row by 6 column array
sumEveryOther = np.array([[0, 0, 1, 0, 1, 0],[0,1,0,1,0,1]])

# Create a multi-dimensional array that when multiplied (using .dot) against vector, results in a
# three element array where the first element is the sum of the 0 and 3 indexed elements of vector,
# the second element is the sum of the 1 and 4 indexed elements of vector, and the third element is
# the sum of the 2 and 5 indexed elements of vector.
# This should be a 3 row by 6 column array
sumEveryThird = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]])

# Create a multi-dimensional array that can be used to sum the first three elements of vector and
# the last three elements of vector, which returns a two element array with those values when dotted
# with vector.
# This should be a 2 row by 6 column array
sumByThree = np.array([[1,1,1,0,0,0],[0,0,0,1,1,1]])

# Create a multi-dimensional array that sums the first two elements, second two elements, and
# last two elements of vector, which returns a three element array with those values when dotted
# with vector.
# This should be a 3 row by 6 column array
sumByTwo = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])

print 'sumEveryOther.dot(vector):\t{0}'.format(sumEveryOther.dot(vector))
print 'sumEveryThird.dot(vector):\t{0}'.format(sumEveryThird.dot(vector))

print '\nsumByThree.dot(vector):\t{0}'.format(sumByThree.dot(vector))
print 'sumByTwo.dot(vector): \t{0}'.format(sumByTwo.dot(vector))

# TEST Aggregation using arrays (4a)
Test.assertEquals(sumEveryOther.shape, (2, 6), 'incorrect shape for sumEveryOther')
Test.assertEquals(sumEveryThird.shape, (3, 6), 'incorrect shape for sumEveryThird')
Test.assertTrue(np.allclose(sumEveryOther.dot(vector), [6, 9]), 'incorrect value for sumEveryOther')
Test.assertTrue(np.allclose(sumEveryThird.dot(vector), [3, 5, 7]),
                'incorrect value for sumEveryThird')
Test.assertEquals(sumByThree.shape, (2, 6), 'incorrect shape for sumByThree')
Test.assertEquals(sumByTwo.shape, (3, 6), 'incorrect shape for sumByTwo')
Test.assertTrue(np.allclose(sumByThree.dot(vector),  [3, 12]), 'incorrect value for sumByThree')
Test.assertTrue(np.allclose(sumByTwo.dot(vector), [1, 5, 9]), 'incorrect value for sumByTwo')

# Reference for what to recreate
print 'sumEveryOther: \n{0}'.format(sumEveryOther)
print '\nsumEveryThird: \n{0}'.format(sumEveryThird)

# TODO: Replace <FILL IN> with appropriate code
# Use np.tile and np.eye to recreate the arrays
sumEveryOtherTile =  np.tile(np.eye(2),3)
sumEveryThirdTile = np.tile(np.eye(3),2)

print sumEveryOtherTile
print 'sumEveryOtherTile.dot(vector): {0}'.format(sumEveryOtherTile.dot(vector))
print '\n', sumEveryThirdTile
print 'sumEveryThirdTile.dot(vector): {0}'.format(sumEveryThirdTile.dot(vector))

# TEST Recreate with `np.tile` and `np.eye` (4b)
Test.assertEquals(sumEveryOtherTile.shape, (2, 6), 'incorrect shape for sumEveryOtherTile')
Test.assertEquals(sumEveryThirdTile.shape, (3, 6), 'incorrect shape for sumEveryThirdTile')
Test.assertTrue(np.allclose(sumEveryOtherTile.dot(vector), [6, 9]),
                'incorrect value for sumEveryOtherTile')
Test.assertTrue(np.allclose(sumEveryThirdTile.dot(vector), [3, 5, 7]),
                'incorrect value for sumEveryThirdTile')

# Reference for what to recreate
print 'sumByThree: \n{0}'.format(sumByThree)
print '\nsumByTwo: \n{0}'.format(sumByTwo)

# TODO: Replace <FILL IN> with appropriate code
# Use np.kron, np.eye, and np.ones to recreate the arrays
sumByThreeKron = np.kron(np.eye(2), np.ones(3))
sumByTwoKron = np.kron(np.eye(3), np.ones(2))

print sumByThreeKron
print 'sumByThreeKron.dot(vector): {0}'.format(sumByThreeKron.dot(vector))
print '\n', sumByTwoKron
print 'sumByTwoKron.dot(vector): {0}'.format(sumByTwoKron.dot(vector))

# TEST Recreate with `np.kron` (4c)
Test.assertEquals(sumByThreeKron.shape, (2, 6), 'incorrect shape for sumByThreeKron')
Test.assertEquals(sumByTwoKron.shape, (3, 6), 'incorrect shape for sumByTwoKron')
Test.assertTrue(np.allclose(sumByThreeKron.dot(vector),  [3, 12]),
                'incorrect value for sumByThreeKron')
Test.assertTrue(np.allclose(sumByTwoKron.dot(vector), [1, 5, 9]),
                'incorrect value for sumByTwoKron')

# TODO: Replace <FILL IN> with appropriate code
# Create a multi-dimensional array to perform the aggregation
T = np.tile(np.eye(20),12)
print format(T.shape)
# Transform scaledData using T.  Make sure to retain the keys.
timeData = scaledData.map(lambda x: (x[0], T.dot(x[1])))

timeData.cache()
print timeData.count()
print timeData.first()

# TEST Aggregate by time (4d)
Test.assertEquals(T.shape, (20, 240), 'incorrect shape for T')
timeDataFirst = timeData.values().first()
timeDataFifth = timeData.values().take(5)[4]
Test.assertEquals(timeData.count(), 46460, 'incorrect length of timeData')
Test.assertEquals(timeDataFirst.size, 20, 'incorrect value length of timeData')
Test.assertEquals(timeData.keys().first(), (0, 0), 'incorrect keys in timeData')
Test.assertTrue(np.allclose(timeDataFirst[:2], [0.00802155, 0.00607693]),
                'incorrect values in timeData')
Test.assertTrue(np.allclose(timeDataFifth[-2:],[-0.00636676, -0.0179427]),
                'incorrect values in timeData')

# TODO: Replace <FILL IN> with appropriate code
componentsTime, timeScores, eigenvaluesTime = pca(timeData.map(lambda x: x[1]), 3)

print 'componentsTime: (first five) \n{0}'.format(componentsTime[:5,:])
print ('\ntimeScores (first three): \n{0}'
       .format('\n'.join(map(str, timeScores.take(3)))))
print '\neigenvaluesTime: (first five) \n{0}'.format(eigenvaluesTime[:5])

# TEST Obtain a compact representation (4e)
Test.assertEquals(componentsTime.shape, (20, 3), 'incorrect shape for componentsTime')
Test.assertTrue(np.allclose(np.abs(np.sum(componentsTime[:5, :])), 2.37299020),
                'incorrect value for componentsTime')
Test.assertTrue(np.allclose(np.abs(np.sum(timeScores.take(3))), 0.0213119114),
                'incorrect value for timeScores')
Test.assertTrue(np.allclose(np.sum(eigenvaluesTime[:5]), 0.844764792),
                'incorrect value for eigenvaluesTime')

scoresTime = np.vstack(timeScores.collect())
imageOneTime = scoresTime[:,0].reshape(230, 202).T
imageTwoTime = scoresTime[:,1].reshape(230, 202).T
brainmap = polarTransform(3, [imageOneTime, imageTwoTime])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap,interpolation='nearest', aspect='auto')
pass

# TODO: Replace <FILL IN> with appropriate code
# Create a multi-dimensional array to perform the aggregation
D = np.kron(np.eye(12), np.ones(20))

print format(D.shape)
# Transform scaledData using D.  Make sure to retain the keys.
directionData = scaledData.map(lambda x: (x[0],D.dot(x[1])))

directionData.cache()
print directionData.count()
print directionData.first()

# TEST Aggregate by direction (4f)
Test.assertEquals(D.shape, (12, 240), 'incorrect shape for D')
directionDataFirst = directionData.values().first()
directionDataFifth = directionData.values().take(5)[4]
Test.assertEquals(directionData.count(), 46460, 'incorrect length of directionData')
Test.assertEquals(directionDataFirst.size, 12, 'incorrect value length of directionData')
Test.assertEquals(directionData.keys().first(), (0, 0), 'incorrect keys in directionData')
Test.assertTrue(np.allclose(directionDataFirst[:2], [ 0.03346365,  0.03638058]),
                'incorrect values in directionData')
Test.assertTrue(np.allclose(directionDataFifth[:2], [ 0.01479147, -0.02090099]),
                'incorrect values in directionData')

# TODO: Replace <FILL IN> with appropriate code
componentsDirection, directionScores, eigenvaluesDirection = pca(directionData.map(lambda x: x[1]), 3)

print 'componentsDirection: (first five) \n{0}'.format(componentsDirection[:5,:])
print ('\ndirectionScores (first three): \n{0}'
       .format('\n'.join(map(str, directionScores.take(3)))))
print '\neigenvaluesDirection: (first five) \n{0}'.format(eigenvaluesDirection[:5])

# TEST Compact representation of direction data (4g)
Test.assertEquals(componentsDirection.shape, (12, 3), 'incorrect shape for componentsDirection')
Test.assertTrue(np.allclose(np.abs(np.sum(componentsDirection[:5, :])), 1.080232069),
                'incorrect value for componentsDirection')
Test.assertTrue(np.allclose(np.abs(np.sum(directionScores.take(3))), 0.10993162084),
                'incorrect value for directionScores')
Test.assertTrue(np.allclose(np.sum(eigenvaluesDirection[:5]), 2.0089720377),
                'incorrect value for eigenvaluesDirection')

scoresDirection = np.vstack(directionScores.collect())
imageOneDirection = scoresDirection[:,0].reshape(230, 202).T
imageTwoDirection = scoresDirection[:,1].reshape(230, 202).T
brainmap = polarTransform(2, [imageOneDirection, imageTwoDirection])
# with thunder: Colorize(cmap='polar', scale=2).transform([imageOneDirection, imageTwoDirection])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap, interpolation='nearest', aspect='auto')
pass


