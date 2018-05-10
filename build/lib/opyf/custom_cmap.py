'''
NAME
    Custom Colormaps for Matplotlib
PURPOSE
    This program shows how to implement make_cmap which is a function that
    generates a colorbar.  If you want to look at different color schemes,
    check out https://kuler.adobe.com/create.
PROGRAMMER(S)
    Chris Slocum
REVISION HISTORY
    20130411 -- Initial version created
    20140313 -- Small changes made and code posted online
    20140320 -- Added the ability to set the position of each color
'''

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]],
                         bit_rgb[colors[i][3]])
    cdict = {'red':[], 'green':[], 'blue':[], 'alpha':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
        cdict['alpha'].append((pos, color[3], color[3]))
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

#### An example of how to use make_cmap
#import matplotlib.pyplot as plt
#import numpy as np
#
#fig = plt.figure()
#ax = fig.add_subplot(311)
#### Create a list of RGB tuples
#colors = [(255,0,0), (255,255,0), (255,255,255), (0,157,0), (0,0,255)] # This example uses the 8-bit RGB
#### Call the function make_cmap which returns your colormap
#my_cmap = make_cmap(colors, bit=True)
#### Use your colormap
#plt.pcolor(np.random.rand(25,50), cmap=my_cmap)
#plt.colorbar()
#
#ax = fig.add_subplot(312)
#colors = [(1,1,1), (0.5,0,0)] # This example uses the arithmetic RGB
#### If you are only going to use your colormap once you can
#### take out a step.
#plt.pcolor(np.random.rand(25,50), cmap=make_cmap(colors))
#plt.colorbar()

#plt.close('all')
#colors = [(33./256,99./256,66./256,0.1), (0.9,0.9,0.9,1), (0.1,0.4,0.5,1), (0.7,0,0,1), (0,0,0,1)]
#### Create an array or list of positions from 0 to 1.
#position = [0, 0.05,  0.6, 0.8, 1]
#cmap=make_cmap(colors, position=position)
#plt.imshow(np.random.rand(25,50), cmap=cmap)
#plt.colorbar()
#plt.figure()
##,norm=mpl.colors.LogNorm(vmin=0.001, vmax=0.5)
#plt.imshow(grid_valr, origin='upper',alpha=0.9,cmap=cmap,vmin=0.01, vmax=0.4)
#plt.colorbar()
#
#plt.savefig("custom_cmap.png")
#plt.show()
