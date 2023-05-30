'''
NAME
    Custom Colormaps for Matplotlib
PURPOSE
    This program shows how to implement make_cmap which is a function that
    generates a colorbar.  If you want to look at different color schemes,
    check out https://kuler.adobe.com/create.
PROGRAMMER(S)
    Chris Slocum
    Gauthier Rousseau
REVISION HISTORY
    20130411 -- Initial version created
    20140313 -- Small changes made and code posted online
    20140320 -- Added the ability to set the position of each color
    20180531 -- Added the alpha (transprency) + different customized color maps

'''
import numpy as np


def make_cmap(colors, position=None, bit=False, res=256):
    '''
    make_cmap takes a list of tuples which contain RGBA values. The RGBA
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl

    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
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
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
        cdict['alpha'].append((pos, color[3], color[3]))
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, res)
    return cmap


# An example of how to use make_cmap
#import matplotlib.pyplot as plt
#import numpy as np
#
#fig = plt.figure()
#ax = fig.add_subplot(311)
# Create a list of RGB tuples
# colors = [(255,0,0), (255,255,0), (255,255,255), (0,157,0), (0,0,255)] # This example uses the 8-bit RGB
# Call the function make_cmap which returns your colormap
#my_cmap = make_cmap(colors, bit=True)
# Use your colormap
#plt.pcolor(np.random.rand(25,50), cmap=my_cmap)
# plt.colorbar()
#
#ax = fig.add_subplot(312)
# colors = [(1,1,1), (0.5,0,0)] # This example uses the arithmetic RGB
# If you are only going to use your colormap once you can
# take out a step.
#plt.pcolor(np.random.rand(25,50), cmap=make_cmap(colors))
# plt.colorbar()

# plt.close('all')
#colors = [(33./256,99./256,66./256,0.1), (0.9,0.9,0.9,1), (0.1,0.4,0.5,1), (0.7,0,0,1), (0,0,0,1)]
# Create an array or list of positions from 0 to 1.
#position = [0, 0.05,  0.6, 0.8, 1]
#cmap=make_cmap(colors, position=position)
#plt.imshow(np.random.rand(25,50), cmap=cmap)
# plt.colorbar()
# plt.figure()
# ,norm=mpl.colors.LogNorm(vmin=0.001, vmax=0.5)
#plt.imshow(grid_valr, origin='upper',alpha=0.9,cmap=cmap,vmin=0.01, vmax=0.4)
# plt.colorbar()
#
# plt.savefig("custom_cmap.png")
# plt.show()


def setcolorRGB(r, g, b, alpha=1., brightness=1):
    r = np.min([255, r*brightness])
    g = np.min([255, g*brightness])
    b = np.min([255, b*brightness])
    return (r/255, g/255, b/255, alpha)
# %%


def make_cmap_customized(Palette='mountain', position=[0.0, 0.16, 0.2, 0.24, 0.4, 0.7, 0.8, 1], invert=False, alpha=1, brightness=1):
    if Palette == 'sunrise':
        couleur7 = setcolorRGB(0, 0, 0, alpha=alpha, brightness=brightness)
        couleur6 = setcolorRGB(64, 50, 79, alpha=alpha, brightness=brightness)
        couleur5 = setcolorRGB(107, 64, 110, alpha=alpha,
                               brightness=brightness)
        couleur4 = setcolorRGB(141, 76, 125, alpha=alpha,
                               brightness=brightness)
        couleur3 = setcolorRGB(172, 85, 122, alpha=alpha,
                               brightness=brightness)
        couleur2 = setcolorRGB(
            210, 124, 124, alpha=alpha, brightness=brightness)
        couleur1 = setcolorRGB(
            240, 206, 125, alpha=alpha, brightness=brightness)
        couleur0 = setcolorRGB(
            255, 255, 255, alpha=alpha, brightness=brightness)
    elif Palette == 'green':
        couleur7 = setcolorRGB(0, 0, 0, alpha=alpha, brightness=brightness)
        couleur6 = setcolorRGB(6, 49, 50, alpha=alpha, brightness=brightness)
        couleur5 = setcolorRGB(28, 78, 78, alpha=alpha, brightness=brightness)
        couleur4 = setcolorRGB(55, 140, 129, alpha=alpha,
                               brightness=brightness)
        couleur3 = setcolorRGB(
            172, 185, 153, alpha=alpha, brightness=brightness)
        couleur2 = setcolorRGB(
            199, 205, 181, alpha=alpha, brightness=brightness)
        couleur1 = setcolorRGB(
            232, 219, 194, alpha=alpha, brightness=brightness)
        couleur0 = setcolorRGB(
            255, 255, 255, alpha=alpha, brightness=brightness)
    elif Palette == 'mountain':
        couleur7 = setcolorRGB(0, 0, 0, alpha=alpha, brightness=brightness)
        couleur6 = setcolorRGB(45, 52, 70, alpha=alpha, brightness=brightness)
        couleur5 = setcolorRGB(89, 76, 96, alpha=alpha, brightness=brightness)
        couleur4 = setcolorRGB(
            145, 101, 118, alpha=alpha, brightness=brightness)
        couleur3 = setcolorRGB(
            212, 119, 127, alpha=alpha, brightness=brightness)
        couleur2 = setcolorRGB(
            212, 153, 154, alpha=alpha, brightness=brightness)
        couleur1 = setcolorRGB(
            238, 189, 184, alpha=alpha, brightness=brightness)
        couleur0 = setcolorRGB(
            255, 255, 255, alpha=alpha, brightness=brightness)
    elif Palette == 'prune':
        couleur7 = setcolorRGB(0, 0, 0, alpha=alpha, brightness=brightness)
        couleur6 = setcolorRGB(66, 37, 67, alpha=alpha, brightness=brightness)
        couleur5 = setcolorRGB(125, 58, 91, alpha=alpha, brightness=brightness)
        couleur4 = setcolorRGB(107, 77, 131, alpha=alpha,
                               brightness=brightness)
        couleur3 = setcolorRGB(
            205, 179, 214, alpha=alpha, brightness=brightness)
        couleur2 = setcolorRGB(
            164, 173, 154, alpha=alpha, brightness=brightness)
        couleur1 = setcolorRGB(
            207, 213, 199, alpha=alpha, brightness=brightness)
        couleur0 = setcolorRGB(
            255, 255, 255, alpha=alpha, brightness=brightness)
    elif Palette == 'asym_mountain5':
        couleur7 = setcolorRGB(45, 52, 70, alpha=alpha, brightness=brightness)
        couleur6 = setcolorRGB(110, 86, 96, alpha=alpha, brightness=brightness)
        couleur5 = setcolorRGB(135, 90, 115, alpha=alpha,
                               brightness=brightness)
        couleur4 = setcolorRGB(
            145, 101, 118, alpha=alpha, brightness=brightness)
        couleur3 = setcolorRGB(
            212, 119, 127, alpha=alpha, brightness=brightness)
        couleur2 = setcolorRGB(
            232, 219, 194, alpha=alpha, brightness=brightness)
        couleur1 = setcolorRGB(
            167, 213, 229, alpha=alpha, brightness=brightness)
        couleur0 = setcolorRGB(
            121, 175, 204, alpha=alpha, brightness=brightness)

    colors = [couleur0, couleur1, couleur2, couleur3,
              couleur4, couleur5, couleur6, couleur7]
    if invert == True:
        colors = np.flipud(colors)
    cmap = make_cmap(colors, position=position, res=1000)
    return cmap


# %%
def make_cmap_customized_asym(Palette='asym_mountain_full', ratio=0.2, param=2, posR=1, posL=1, paramR=1, paramL=1, lightL=1., invert=False, brightness=1):

    if Palette == 'asym_mountain_full':
        couleur10 = setcolorRGB(45, 52, 70, brightness=brightness)
        couleur9 = setcolorRGB(145, 101, 118, brightness=brightness)
        couleur8 = setcolorRGB(212, 119, 127, brightness=brightness)
        couleur7 = setcolorRGB(213, 153, 153, brightness=brightness)
        couleur6 = setcolorRGB(232, 172, 171, brightness=brightness)
        couleur5 = setcolorRGB(232, 219, 194, brightness=brightness)
        couleur4 = setcolorRGB(167, 213, 229, brightness=brightness)
        couleur3 = setcolorRGB(121, 175, 204, brightness=brightness)
        couleur2 = setcolorRGB(36, 140, 163, brightness=brightness)
        couleur1 = setcolorRGB(0, 70, 135, brightness=brightness)
        couleur0 = setcolorRGB(6, 20, 31, brightness=brightness)

    if Palette == 'asym_mountain_green':
        couleur10 = setcolorRGB(45, 52, 70, brightness=brightness)
        couleur9 = setcolorRGB(145, 101, 118, brightness=brightness)
        couleur8 = setcolorRGB(212, 119, 127, brightness=brightness)
        couleur7 = setcolorRGB(213, 153, 153, brightness=brightness)
        couleur6 = setcolorRGB(232, 172, 171, brightness=brightness)
        couleur5 = setcolorRGB(188, 236, 202, brightness=brightness)
        couleur4 = setcolorRGB(167, 213, 229, brightness=brightness)
        couleur3 = setcolorRGB(121, 175, 204, brightness=brightness)
        couleur2 = setcolorRGB(36, 140, 163, brightness=brightness)
        couleur1 = setcolorRGB(0, 70, 135, brightness=brightness)
        couleur0 = setcolorRGB(6, 20, 31, brightness=brightness)

    position = np.linspace(0, 1, 11)

    position = 0.5+0.5*np.absolute(((position-0.5)/0.5))**param
    position[0:5] = 1-position[0:5]

    position = position.tolist()
    colors = [couleur0, couleur1, couleur2, couleur3, couleur4,
              couleur5, couleur6, couleur7, couleur8, couleur9, couleur10]

    if invert == True:
        colors = np.flipud(colors)
    cmap = make_cmap(colors, position=position, res=1000)
    # Rebluid a color vector to scale with the scalar
    posL = np.array([0, 0.2, 0.4, 0.6, 0.8])
    posL2 = (posL)**paramL*ratio
    posR = np.array([0, 0.2, 0.4, 0.6, 0.8])
    posR2 = np.flip(1-(posR)**paramR*(1-ratio), axis=0)
    position2 = np.concatenate([posL2, [ratio], posR2])
    position2 = position2.tolist()
    # find the maximum
    if ratio < 0.5:
        colorsR = cmap(np.linspace(0.5, 1, 6))
        colorsL = cmap(np.linspace(0.5-ratio/(1-ratio)*0.5, 0.5, 6))
    if ratio >= 0.5:
        colorsL = cmap(np.linspace(0, 0.5, 6))
        colorsR = cmap(np.linspace(0.5, 0.5+(1-ratio)/(ratio)*0.5, 6))

    colors = np.concatenate([colorsL[:-1], colorsR]).tolist()
    cmap = make_cmap(colors, position=position2, res=1000)
    return cmap
