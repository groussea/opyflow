#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:39:27 2017

@author: Gauthier ROUSSEAU
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib as mpl
import cv2


#fig,ax=opyffigureandaxes(scalefig,vis[:,:,0],'New',setPlot['unit'])
#ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),extent=[0,Lvis*setPlot['scale'],Hvis*setPlot['scale'],0],zorder=0) 
##  
    
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor']='0'
plt.rcParams['axes.linewidth']=1.5
#plt.rc('text', usetex=True) may be activated if you prefer a Latex rendering font
#plt.rc('font', family='arial')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

plt.close('all')
plt.rc('font', size=10)
def ir(A):
    return int(np.round(A))

def opyfFigure():
    fig=plt.figure()
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(2000,200,1200, 900)
    return fig
    

def opyfPlot(grid_x,grid_y,gridVx,gridVy,Xdata,Vdata,setPlot,vis=None,Ptype='norme',namefig='Opyf',vlim=None,scale=None,cmap=None,alpha=0.6,width=0.002,nvec=3000,respx=32,ROIvis=None,**args):

    if cmap is None:
        cmap=setcmap(Ptype,alpha=alpha)
        
    normalize=args.get('normalize',False)
    extentr=args.get('extentr',setPlot['extentFrame'])
    infoPlotQuiver={'cmap':cmap,
          'width' :width,
          'alpha':alpha,
          'vlim':vlim,
          'scale':scale} 

    infoPlotPointCloud={'cmap':cmap,
          'alpha':alpha,
          'vlim':vlim}
    infoPlotField={'cmap': cmap,                    
                    'vlim':vlim}
    

        
 
    fig,ax=opyffigureandaxes(extent=setPlot['extentFrame'],Hfig=9,unit=setPlot['unit'],num=namefig)
    if setPlot['DisplayVis']==True and vis is not None:  
        if ROIvis is not None:
            vis=vis[ROIvis[1]:(ROIvis[3]+ROIvis[1]),ROIvis[0]:(ROIvis[2]+ROIvis[0])]
        vis =CLAHEbrightness(vis,0)  
        if len(np.shape(vis))==3:           
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),extent=extentr,zorder=0) 
        else:
            ax.imshow(vis,extent=extentr,zorder=0,cmap=mpl.cm.gray) 
 
    Field=setField(gridVx,gridVy,Ptype) 
    
    
    
    if setPlot['DisplayField']==True:
        resx=np.absolute(grid_x[0,1]-grid_x[0,0])
        resy=np.absolute(grid_y[1,0]-grid_y[0,0])
        extent=[grid_x[0,0]-resx/2,grid_x[0,-1]+resx/2,grid_y[-1,0]-resy/2,grid_y[0,0]+resy/2]
#        figp,ax,im=opyfField2(grid_x,grid_y,Field,ax=ax,**infoPlotField)
        fig,ax,im=opyfField(Field,ax=ax,extent=extent,extentr=extentr,**infoPlotField)

   
        fig,cb=opyfColorBar(fig,im,label=Ptype+' velocity (in '+setPlot['unit']+'/DeltaT)')

    if setPlot['QuiverOnFieldColored']==True:
        figp,ax,qv,sm=opyfQuiverFieldColored(grid_x,grid_y,gridVx,gridVy,ax=ax,res=respx,normalize=normalize,**infoPlotQuiver)
        figp,cb=opyfColorBar(fig,sm,label=Ptype+' velocity (in '+setPlot['unit']+'/DeltaT)')

    if setPlot['QuiverOnField']==True:
        figp,ax,qv=opyfQuiverField(grid_x,grid_y,gridVx,gridVy,ax=ax,res=respx,normalize=normalize,**infoPlotQuiver)


    if setPlot['DisplayPointsColored']==True:
        figp,ax,sc=opyfPointCloudColoredScatter(Xdata,Vdata,ax=ax,s=10,cmapCS=cmap,**infoPlotPointCloud)
        figp,cb=opyfColorBar(fig,sc,label=Ptype+' velocity (in '+setPlot['unit']+'/DeltaT)')
        cb.set_alpha(0.8)
        cb.draw_all()

    if setPlot['DisplayPoints']==True:
        figp,ax=opyfPointCloudScatter(Xdata,Vdata,ax=ax,s=10,color='k',**infoPlotPointCloud)

        
 
    if setPlot['QuiverOnPoints']==True:
       figp,ax,qv=opyfQuiverPointCloud(Xdata,Vdata,ax=ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)

    if setPlot['QuiverOnPointsColored']==True:
       figp,ax,qv,sm=opyfQuiverPointCloudColored(Xdata,Vdata,ax=ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)
       figp,cb=opyfColorBar(fig,sm,label='Amplitude (in '+setPlot['unit']+'/DeltaT)')
       cb.set_alpha(0.8)
       cb.draw_all()

    return fig,ax

def opyfPlotRectilinear(vecX,vecY,gridVx,gridVy,setPlot,Xdata=None,Vdata=None,vis=None,Hfig=9,Ptype='norme',namefig='Opyf',vlim=None,scale=None,cmap=None,alpha=0.6,width=0.002,nvec=3000,res=32,ROIvis=None,**args):
    
    grid_x = np.ones((len(vecY),1)) * vecX
    grid_y = (np.ones((len(vecX),1)) * vecY).T
    if cmap is None:
        cmap=setcmap(Ptype,alpha=alpha)
        
    normalize=args.get('normalize',False)
    extentr=args.get('extentr',setPlot['extentFrame'])
    infoPlotQuiver={'cmap':cmap,
          'width' :width,
          'alpha':alpha,
          'vlim':vlim,
          'scale':scale} 

    infoPlotPointCloud={'cmap':cmap,
          'alpha':alpha,
          'vlim':vlim}
    infoPlotField={'cmap': cmap,                    
                    'vlim':vlim}
    

        
 
    fig,ax=opyffigureandaxes(extent=setPlot['extentFrame'],Hfig=Hfig,unit=setPlot['unit'][0],num=namefig)
    if setPlot['DisplayVis']==True and vis is not None:  
        if ROIvis is not None:
            vis=vis[ROIvis[1]:(ROIvis[3]+ROIvis[1]),ROIvis[0]:(ROIvis[2]+ROIvis[0])]
        vis =CLAHEbrightness(vis,0)  
        if len(np.shape(vis))==3:           
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),extent=extentr,zorder=0) 
        else:
            ax.imshow(vis,extent=extentr,zorder=0,cmap=mpl.cm.gray) 
 
    Field=setField(gridVx,gridVy,Ptype) 
    
    
    
    if setPlot['DisplayField']==True:
        resx=np.absolute(grid_x[0,1]-grid_x[0,0])
        resy=np.absolute(grid_y[1,0]-grid_y[0,0])
        extent=[grid_x[0,0]-resx/2,grid_x[0,-1]+resx/2,grid_y[-1,0]-resy/2,grid_y[0,0]+resy/2]
#        figp,ax,im=opyfField2(grid_x,grid_y,Field,ax=ax,**infoPlotField)
        figp,ax,im=opyfField(Field,ax=ax,extent=extent,extentr=extentr,**infoPlotField)

   
        figp,cb=opyfColorBar(fig,im,label=Ptype+' velocity (in '+setPlot['unit'][0]+'/'+setPlot['unit'][1] +')')

    if setPlot['QuiverOnFieldColored']==True:
        figp,ax,qv,sm=opyfQuiverFieldColored(grid_x,grid_y,gridVx,gridVy,ax=ax,res=res,normalize=normalize,**infoPlotQuiver)
        figp,cb=opyfColorBar(fig,sm,label=Ptype+' velocity (in '+setPlot['unit'][0]+'/'+setPlot['unit'][1]+ ')')

    if setPlot['QuiverOnField']==True:
        figp,ax,qv=opyfQuiverField(grid_x,grid_y,gridVx,gridVy,ax=ax,res=res,normalize=normalize,**infoPlotQuiver)


    if setPlot['DisplayPointsColored']==True and Xdata is not None and Vdata is not None:
        figp,ax,sc=opyfPointCloudColoredScatter(Xdata,Vdata,ax=ax,s=10,cmapCS=cmap,**infoPlotPointCloud)
        figp,cb=opyfColorBar(fig,sc,label=Ptype+' velocity (in '+setPlot['unit'][0]+'/'+setPlot['unit'][1]+ ')')
        cb.set_alpha(0.8)
        cb.draw_all()
#
    if setPlot['DisplayPoints']==True and Xdata is not None and Vdata is not None:
        figp,ax=opyfPointCloudScatter(Xdata,Vdata,ax=ax,s=10,color='k',**infoPlotPointCloud)

        
 
    if setPlot['QuiverOnPoints']==True and Xdata is not None and Vdata is not None:
       figp,ax,qv=opyfQuiverPointCloud(Xdata,Vdata,ax=ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)

    if setPlot['QuiverOnPointsColored']==True and Xdata is not None and Vdata is not None:
       figp,ax,qv,sc=opyfQuiverPointCloudColored(Xdata,Vdata,ax=ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)
       figp,cb=opyfColorBar(fig,sc,label=Ptype+' velocity (in '+setPlot['unit'][0]+'/'+setPlot['unit'][1]+ ')')
       cb.set_alpha(0.8)
       cb.draw_all()

    return fig,ax

class opyfDisplayer:   
    def __init__(self,**setPlot): 
        self.setPlot={'DisplayVis':setPlot.get('DisplayVis',False), # If the frame availaible
             'DisplayField':setPlot.get('DisplayField',False),
             'QuiverOnFieldColored':setPlot.get('QuiverOnFieldColored',False),
             'QuiverOnField':setPlot.get('QuiverOnField',False),
             'DisplayPointsColored':setPlot.get('DisplayPointsColored',False),
             'DisplayPoints':setPlot.get('DisplayPoints',False),
             'QuiverOnPointsColored':setPlot.get('QuiverOnPointsColored',False),
             'QuiverOnPoints':setPlot.get('QuiverOnPoints',False),
             'ScaleVectors':setPlot.get('ScaleVectors',0.1),
             'vecX':setPlot.get('vecX',[]),
             'vecY':setPlot.get('vecY',[]),
             'extentFrame':setPlot.get('extentFrame',[0,1,0,1]), 
             'unit':setPlot.get('unit',['','deltaT']),
             'Hfig':setPlot.get('Hfig',9),
             'num':setPlot.get('num','opyfPlot'),
             'grid':setPlot.get('grid',True),
             'vlim':setPlot.get('vlim',None)} 


        if (self.setPlot['DisplayField']==True or
        self.setPlot['QuiverOnFieldColored']==True or
        self.setPlot['QuiverOnField']==True): 
            if (len(self.setPlot['vecX'])>0 and len(self.setPlot['vecY'])>0):
                self.setPlot['type']='Rectilinear'
                self.grid_x = np.ones((len(setPlot['vecY']),1)) * setPlot['vecX']
                self.grid_y = (np.ones((len(setPlot['vecX']),1)) * setPlot['vecY']).T
                if self.setPlot['extentFrame']==[0,1,0,1]:
                    self.setPlot['extentFrame']=[setPlot['vecX'][0],setPlot['vecX'][-1],setPlot['vecY'][-1],setPlot['vecY'][0]]
            else:
                print('provide vecX and vecY arguments to initialize a rectilinear plot')
        plt.ioff()     
        
        self.fig,self.ax= opyffigureandaxes(extent=self.setPlot['extentFrame'],Hfig=self.setPlot['Hfig'],unit=self.setPlot['unit'][0],num='opyfPlot')
           
    def plot(self,gridVx=None,gridVy=None,Xdata=None,Vdata=None,vis=None,Hfig=9,Ptype='norme',namefig='Opyf',scale=None,cmap=None,alpha=0.6,width=0.002,nvec=3000,res=32,ROIvis=None,**args):
        
        if len(self.ax.collections)>0:
            del self.ax.collections[:]
        if len(self.ax.lines)>0:   
            del self.ax.lines[:]
        if len(self.ax.images)>0:   
            del self.ax.images[:]
        if len(self.fig.axes)>1:
            del self.fig.axes[1].collections[:]
#            self.fig,self.ax= opyffigureandaxes(extent=self.setPlot['extentFrame'],Hfig=self.setPlot['Hfig'],unit=self.setPlot['unit'][0],num='opfPlot')

        if cmap is None:
            cmap=setcmap(Ptype,alpha=alpha)
            
        normalize=args.get('normalize',False)
        extentr=args.get('extentr',self.setPlot['extentFrame'])
        vlim=args.get('vlim',self.setPlot['vlim'])
        infoPlotQuiver={'cmap':cmap,
              'width' :width,
              'alpha':alpha,
              'vlim':vlim,
              'scale':scale} 
    
        infoPlotPointCloud={'cmap':cmap,
              'alpha':alpha,
              'vlim':vlim}
        infoPlotField={'cmap': cmap,                    
                        'vlim':vlim}
        
    
            
            
        
        if self.setPlot['DisplayVis']==True and vis is not None:  
            if ROIvis is not None:
                vis=vis[ROIvis[1]:(ROIvis[3]+ROIvis[1]),ROIvis[0]:(ROIvis[2]+ROIvis[0])]
            vis =CLAHEbrightness(vis,0)  
            if len(np.shape(vis))==3:           
                self.ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),extent=extentr,zorder=0) 
            else:
                self.ax.imshow(vis,extent=extentr,zorder=0,cmap=mpl.cm.gray) 
        if  gridVx is not None and gridVy is not None:
            Field=setField(gridVx,gridVy,Ptype) 
            
            
            
        if self.setPlot['DisplayField']==True:
            resx=np.absolute(self.grid_x[0,1]-self.grid_x[0,0])
            resy=np.absolute(self.grid_y[1,0]-self.grid_y[0,0])
            extent=[self.grid_x[0,0]-resx/2,self.grid_x[0,-1]+resx/2,self.grid_y[-1,0]-resy/2,self.grid_y[0,0]+resy/2]
    #        figp,ax,im=opyfField2(grid_x,grid_y,Field,ax=ax,**infoPlotField)
            self.fig,self.ax,self.im=opyfField(Field,ax=self.ax,extent=extent,extentr=extentr,**infoPlotField)
    
       
            self.fig,self.cb=opyfColorBar(self.fig,self.im,label=Ptype+' velocity (in '+self.setPlot['unit'][0]+'/'+self.setPlot['unit'][1] +')')
    
        if self.setPlot['QuiverOnFieldColored']==True:
            self.fig,self.ax,self.qv,self.sm=opyfQuiverFieldColored(self.grid_x,self.grid_y,gridVx,gridVy,ax=self.ax,res=res,normalize=normalize,**infoPlotQuiver)
            self.fig,cb=opyfColorBar(self.fig,self.sm,label=Ptype+' velocity (in '+self.setPlot['unit'][0]+'/'+self.setPlot['unit'][1]+ ')')
    
        if self.setPlot['QuiverOnField']==True:
            self.fig,self.ax,qv=opyfQuiverField(self.grid_x,self.grid_y,gridVx,gridVy,ax=self.ax,res=res,normalize=normalize,**infoPlotQuiver)
    
    
        if self.setPlot['DisplayPointsColored']==True and Xdata is not None and Vdata is not None:
            self.fig,self.ax,self.sc=opyfPointCloudColoredScatter(Xdata,Vdata,ax=self.ax,s=10,cmapCS=cmap,**infoPlotPointCloud)
            self.fig,self.cb=opyfColorBar(self.fig,self.sc,label=Ptype+' velocity (in '+self.setPlot['unit'][0]+'/'+self.setPlot['unit'][1]+ ')')
            self.cb.set_alpha(0.8)
            self.cb.draw_all()
    #
        if self.setPlot['DisplayPoints']==True and Xdata is not None and Vdata is not None:
            self.fig,self.ax=opyfPointCloudScatter(Xdata,Vdata,ax=self.ax,s=10,color='k',**infoPlotPointCloud)
    
            
     
        if self.setPlot['QuiverOnPoints']==True and Xdata is not None and Vdata is not None:
           self.fig,self.ax,qv=opyfQuiverPointCloud(Xdata,Vdata,ax=self.ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)
    
        if self.setPlot['QuiverOnPointsColored']==True and Xdata is not None and Vdata is not None:
           self.fig,self.ax,self.qv,sc=opyfQuiverPointCloudColored(Xdata,Vdata,ax=self.ax,nvec=nvec,normalize=normalize,**infoPlotQuiver)
           self.fig,cb=opyfColorBar(self.fig,sc,label=Ptype+' velocity (in '+self.setPlot['unit'][0]+'/'+self.setPlot['unit'][1]+ ')')
           self.cb.set_alpha(0.8)
           self.cb.draw_all()
        self.fig.show()




def opyfText(dictionary,ax=None,fig=None,pos=(0.,0.),esp=50,fontsize=12.,addText=None,alpha=0.8):
        fig,ax=getax(fig=fig,ax=ax)
        dech,decv, esp=pos[0],pos[1], esp
        lk_params=dictionary.get('lk_params',None)
        interp_params=dictionary.get('interp_params',None)
        feature_params=dictionary.get('feature_params',None)

        Text='OPyF - Tracking Good Feature with Optical Flow estimation (LKpyr method)'
        if feature_params is not None:
            Text+='\n G Features to T : maxCorners:'+str(feature_params['maxCorners'])
            Text+=', minDist:'+str(feature_params['minDistance'])
            Text+=', quality level:'+str(feature_params['qualityLevel'])

        if lk_params is not None:
            Text+= '\n lkparam : criteria:'+str(lk_params['criteria'])
            Text+=', maxlevel:'+str(lk_params['maxLevel'])
            Text+=', winsize:'+str(lk_params['winSize'])                        
            
        if interp_params is not None:
            Text+='\n interpolator Type:'+interp_params['kernel']+',Radius:'+str(interp_params['Radius']) 
        if addText is not None:
            Text+='\n'
            Text+=addText
            
        ax.text(dech,decv, Text,bbox=dict(facecolor='white', alpha=alpha),fontsize=fontsize)


def opyfQuiverPointCloud(X,V,fig=None,ax=None,nvec=3000,normalize=False,**args):
    
    fig,ax=getax(fig=fig,ax=ax)
    #one over N
    #Select randomly N vectors
    if len(X)<nvec:
        N=len(X)
        print('only '+str(N)+'vectors plotted because length(X) >' + str(nvec))
    else:
        N=nvec
#    scale=args.get('scale',None)
    if 'cmap' in args:
        del args['cmap']
    if 'vlim' in args:
        del args['vlim']
    ind=np.random.choice(np.arange(len(X)), N, replace = False)
    Xc=X[ind,:]
    Vc=V[ind,:]
    if normalize==False:
        qv=ax.quiver(Xc[:,0],Xc[:,1],Vc[:,0],Vc[:,1],**args) 
    else:
        Norme=(Vc[:,0]**2+Vc[:,1]**2)**0.5
        qv=ax.quiver(Xc[:,0],Xc[:,1],Vc[:,0]/Norme,Vc[:,1]/Norme,**args) 
    return fig,ax,qv


def opyfQuiverPointCloudColored(X,V,fig=None,ax=None,nvec=3000,normalize=False,**args):
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm
    cmap=args.get('cmap',mpl.cm.coolwarm)
    del args['cmap']
    fig,ax=getax(fig=fig,ax=ax)
    #one over N
    #Select randomly N vectors
    if len(X)<nvec:
        N=len(X)
        print('only '+str(N)+'vectors plotted because length(X) >' + str(nvec))
    else:
        N=nvec
    
    ind=np.random.choice(np.arange(len(X)), N, replace = False)
    Xc=X[ind,:]
    Vc=V[ind,:]
    colors=(Vc[:,0]**2+Vc[:,1]**2)**0.5
    if 'vlim' in args:
        vlim=args.get('vlim',[colors.min(),colors.max()])
        if vlim is None:
            vlim=[colors.min(),colors.max()]
        del args['vlim']
    else:
        vlim=[colors.min(),colors.max()]
    norm = Normalize()
    norm.autoscale(colors)
    norm.vmin=vlim[0]
    norm.vmax=vlim[1]
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    if normalize == False:
        qv=ax.quiver(Xc[:,0],Xc[:,1],Vc[:,0],Vc[:,1],color=cmap(norm(colors)),**args) 
    else:
        qv=ax.quiver(Xc[:,0],Xc[:,1],Vc[:,0]/colors,Vc[:,1]/colors,color=cmap(norm(colors)),**args) 
        
    return fig,ax,qv,sm


def opyfQuiverFieldColored(grid_x,grid_y,gridVx,gridVy,fig=None,ax=None,res=32,normalize=False,**args):

    fig,ax=getax(fig=fig,ax=ax)
    import opyf
    from matplotlib.colors import Normalize
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm
    cmap=args.get('cmap',mpl.cm.coolwarm)
    del args['cmap']


    #one over N
    #Select randomly N vectors
    l,c=grid_x.shape
    resx=np.absolute(grid_x[0,1]-grid_x[0,0])
    resy=np.absolute(grid_y[1,0]-grid_y[0,0])
    densx=int(np.round(res/resx))
    densy=int(np.round(res/resy))
    lvec=np.arange(densy/2,l-densy/2,densy,dtype=int)+(l % densy)//2
    cvec=np.arange(densx/2,c-densx/2,densx,dtype=int)+(l % densx)//2
    new_grid_x=np.zeros(len(lvec))
    size=(len(lvec),len(cvec))
    temp_grid_x=grid_x[lvec,:]
    new_grid_x=temp_grid_x[:,cvec]
    temp_grid_y=grid_y[lvec,:]
    new_grid_y=temp_grid_y[:,cvec]
    
    new_gridVx=np.zeros(size)
    new_gridVy=np.zeros(size) 

    for i in range(size[0]):
        for j in range(size[1]):
            new_gridVx[i,j]=np.mean(gridVx[lvec[i]-densy//2:lvec[i]+densy//2+1,cvec[j]-densx//2:cvec[j]+densx//2+1])
            new_gridVy[i,j]=np.mean(gridVy[lvec[i]-densy//2:lvec[i]+densy//2+1,cvec[j]-densx//2:cvec[j]+densx//2+1])
     
    TargetPoints=opyf.Interpolate.npGrid2TargetPoint2D(new_grid_x,new_grid_y)
    Velocities=opyf.Interpolate.npGrid2TargetPoint2D(new_gridVx,new_gridVy)    
    Norme=(Velocities[:,0]**2+Velocities[:,1]**2)**0.5
    if 'vlim' in args:
        vlim=args.get('vlim',[Norme.min(),Norme.max()])
        if vlim is None:
            vlim=[Norme.min(),Norme.max()]
        del args['vlim']
    else:
        vlim=[Norme.min(),Norme.max()]

    norm = Normalize()
    norm.autoscale(Norme)
    norm.vmin=vlim[0]
    norm.vmax=vlim[1]
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])    
    if normalize==False:
        qv=ax.quiver(TargetPoints[:,0],TargetPoints[:,1],Velocities[:,0],Velocities[:,1],color=cmap(norm(Norme)),**args) 
    else:
        qv=ax.quiver(TargetPoints[:,0],TargetPoints[:,1],Velocities[:,0]/Norme[:],Velocities[:,1]/Norme[:],color=cmap(norm(Norme)),**args) 
        
        
    return fig,ax,qv,sm

def opyfQuiverFieldColoredScaled(grid_x,grid_y,gridVx,gridVy,fig=None,ax=None,res=32,**args):

    fig,ax=getax(fig=fig,ax=ax)
    import opyf
    from matplotlib.colors import Normalize
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm
    cmap=args.get('cmap',mpl.cm.coolwarm)
    del args['cmap']


    #one over N
    #Select randomly N vectors
    l,c=grid_x.shape
    resx=np.absolute(grid_x[0,1]-grid_x[0,0])
    resy=np.absolute(grid_y[1,0]-grid_y[0,0])
    densx=int(np.round(res/resx))
    densy=int(np.round(res/resy))
    lvec=np.arange(densy/2,l-densy/2,densy,dtype=int)+(l % densy)//2
    cvec=np.arange(densx/2,c-densx/2,densx,dtype=int)+(l % densx)//2
    new_grid_x=np.zeros(len(lvec))
    size=(len(lvec),len(cvec))
    temp_grid_x=grid_x[lvec,:]
    new_grid_x=temp_grid_x[:,cvec]
    temp_grid_y=grid_y[lvec,:]
    new_grid_y=temp_grid_y[:,cvec]
    
    new_gridVx=np.zeros(size)
    new_gridVy=np.zeros(size) 

    for i in range(size[0]):
        for j in range(size[1]):
            new_gridVx[i,j]=np.mean(gridVx[lvec[i]-densy//2:lvec[i]+densy//2+1,cvec[j]-densx//2:cvec[j]+densx//2+1])
            new_gridVy[i,j]=np.mean(gridVy[lvec[i]-densy//2:lvec[i]+densy//2+1,cvec[j]-densx//2:cvec[j]+densx//2+1])
     
    TargetPoints=opyf.Interpolate.npGrid2TargetPoint2D(new_grid_x,new_grid_y)
    Velocities=opyf.Interpolate.npGrid2TargetPoint2D(new_gridVx,new_gridVy)    
    Norme=(Velocities[:,0]**2+Velocities[:,1]**2)**0.5
    if 'vlim' in args:
        vlim=args.get('vlim',[Norme.min(),Norme.max()])
        if vlim is None:
            vlim=[Norme.min(),Norme.max()]
        del args['vlim']
    else:
        vlim=[Norme.min(),Norme.max()]

    norm = Normalize()
    norm.autoscale(Norme)
    norm.vmin=vlim[0]
    norm.vmax=vlim[1]
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])    

    qv=ax.quiver(TargetPoints[:,0],TargetPoints[:,1],Velocities[:,0]/Norme[:],Velocities[:,1]/Norme[:],color=cmap(norm(colors)),**args) 

    return fig,ax,qv,sm


def opyfQuiverField(grid_x,grid_y,gridVx,gridVy,fig=None,ax=None,res=32,normalize=False,**args):
    fig,ax=getax(fig=fig,ax=ax)
    import opyf
    if 'cmap' in args:
        del args['cmap']
    if 'vlim' in args:
        del args['vlim']
    #one over N
    #Select randomly N vectors
    l,c=grid_x.shape
    resx=np.absolute(grid_x[0,1]-grid_x[0,0])
    resy=np.absolute(grid_y[1,0]-grid_y[0,0])
    densx=int(res/resx)
    densy=int(res/resy)
    lvec=np.arange(densy/2,l-densy/2,densy,dtype=int)+(l % densy)//2
    cvec=np.arange(densx/2,c-densx/2,densx,dtype=int)+(l % densx)//2
    new_grid_x=np.zeros(len(lvec))
    size=(len(lvec),len(cvec))
    temp_grid_x=grid_x[lvec,:]
    new_grid_x=temp_grid_x[:,cvec]
    temp_grid_y=grid_y[lvec,:]
    new_grid_y=temp_grid_y[:,cvec]
    
    new_gridVx=np.zeros(size)
    new_gridVy=np.zeros(size) 

    for i in range(size[0]):
        for j in range(size[1]):
            new_gridVx[i,j]=np.mean(gridVx[lvec[i]-densy//2:lvec[i]+densy//2,cvec[j]-densx//2:cvec[j]+densx//2])
            new_gridVy[i,j]=np.mean(gridVy[lvec[i]-densy//2:lvec[i]+densy//2,cvec[j]-densx//2:cvec[j]+densx//2])
     
    TargetPoints=opyf.Interpolate.npGrid2TargetPoint2D(new_grid_x,new_grid_y)
    Velocities=opyf.Interpolate.npGrid2TargetPoint2D(new_gridVx,new_gridVy)    
    colors=(Velocities[:,0]**2+Velocities[:,1]**2)**0.5  
    if normalize==False:
        qv=ax.quiver(TargetPoints[:,0],TargetPoints[:,1],Velocities[:,0],Velocities[:,1],**args) 
    else:
        Norme=(Velocities[:,0]**2+Velocities[:,1]**2)**0.5
        qv=ax.quiver(TargetPoints[:,0],TargetPoints[:,1],Velocities[:,0]/Norme,Velocities[:,1]/Norme,**args) 

    return fig,ax,qv





def opyfPointCloudColoredPatch(X,V,fig=None,ax=None,**args):
    from matplotlib.collections import PatchCollection
    fig,ax=getax(fig=fig,ax=ax,values=X)
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm   
    clim=args.get('vlim',[np.min(V),np.max(V)]) 
    del args['vlim']
    radius=args.get('markersize',4)
    del args['markersize']
    patches=[]
    for [x,y] in X:
        circle = plt.Circle((int(x), int(y)), radius=radius)
        patches.append(circle)
        
    p = PatchCollection(patches,clim=clim,edgecolor='none',**args)    
    p.set_array((V[:,0]**2+V[:,1]**2)**0.5)
    ax.add_collection(p) 
                
    
    return fig,ax

def opyfPointCloudColoredScatter(X,V,fig=None,ax=None,cmapCS=mpl.cm.coolwarm,**args):
    from matplotlib.colors import Normalize
    fig,ax=getax(fig=fig,ax=ax)
    if 'cmap' in args:
        del args['cmap']
    norme=(V[:,0]**2+V[:,1]**2)**0.5
    norm = Normalize()
    norm.autoscale(norme)
    vlim=args.get('vlim',[np.min(norme),np.max(norme)])
    if vlim is None:
        vlim=[np.min(norme),np.max(norme)]
    args['vmin']=vlim[0]
    args['vmax']=vlim[1]
    if 'vlim' in args:
        del args['vlim']
    if 'markersize' in args:
        del args['markersize']

#    sc=ax.scatter(X[:,0], X[:,1],c=norme,color=cmapCS(norm(norme)),**args)          
    sc=ax.scatter(X[:,0], X[:,1],c=norme,cmap=cmapCS,**args)  
    
    return fig,ax,sc

def opyfPointCloudScatter(X,V,fig=None,ax=None,**args):
    from matplotlib.collections import PatchCollection
    fig,ax=getax(fig=fig,ax=ax)
    if 'vlim' in args:
        del args['vlim']
    if 'markersize' in args:
        del args['markersize']
    ax.scatter(X[:,0], X[:,1],**args)          
    
    return fig,ax


def opyfPointCloud(X,fig=None,ax=None,**args):
    if 'vlim' in args:
        del args['vlim']
    if 'cmap' in args:
        del args['cmap']
    fig,ax=getax(fig=fig,ax=ax)  
    ax.plot(X[:,0],X[:,1],'.',**args)
    
    return fig,ax




def opyfColorBar(fig,im,label='Magnitude [px/Dt]',**args):
    cbaxes = fig.add_axes([0.15, 0.1, 0.70, 0.03])
    cb=fig.colorbar(im, cax=cbaxes,orientation='horizontal',**args)
    cb.set_label(label) 
    return fig,cb



    
def opyffigureandaxes(extent=[0,1,0,1],unit='px',Hfig=9,**args):
    Hframe=np.absolute(extent[3]-extent[2])
    Lframe=np.absolute(extent[1]-extent[0])
    Lfig=Lframe*Hfig/Hframe   
    #Location of the plot
#    coefy=0.0004
    exty=0.75
    extx=Lframe*exty*Hfig/Lfig/Hframe
    axiswindow=[(1-extx)/2+0.02, 0.2, extx, exty]
    
    num=args.get('num','opyf-Figure')
    if num=='opyf-Figure':
        plt.close(num)
    fig=plt.figure(figsize=(Lfig, Hfig),**args)
    fig.clf()  
    ax = plt.Axes(fig, axiswindow)
    fig.add_axes(ax)
    plt.ylabel('Y['+unit+']')
    plt.xlabel('X['+unit+']')  
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    
    return fig,ax

def setcmap(Type,alpha=1.):
    from opyf.custom_cmap import make_cmap
    if Type=='norme':
        colors = [(33./255,66./255,99./255,0.),(33./255,66./255,99./255,alpha),(1,1,0.3,alpha), (0.8,0,0,alpha), (0,0,0,alpha)]
        position = [0,0.01,0.05,0.5, 1]   
        cmap=make_cmap(colors, position=position) 
        cmap.set_under('g')
        cmap.set_over('g')
    elif Type=='horizontal' or setPlot['Type']=='vertical':
        colors = [(11./255,22./255,33./255,1),(33./255,66./255,99./255,1),(103./255,230./255,93./255,0.3), (0.7,0,0,1),(1.,0,0,1)]
        position = [0.,0.4, 0.5, 0.6,1.]               
        cmap=make_cmap(colors, position=position)   
        cmap.set_under('g')
        cmap.set_over('g')
    else: 
        cmap=mpl.cm.coolwarm
        
    return cmap

def setField(gridVec_x,gridVec_y,Type):
        
    if Type=='norme':    
        Field=(gridVec_x**2+gridVec_y**2)**(0.5) 
    elif Type=='horizontal':
        Field=gridVec_x
    elif Type=='vertical':
        Field=gridVec_y
    return Field



def getax(fig=None,ax=None,values=None):           
    if fig is None and ax is None:  
        if values is not None:
            [l,c]=values.shape
            if c==2:
                fig,ax=opyffigureandaxes(int(values[:,1].max()),int(values[:,0].max()))
        else:
            fig=plt.figure()
            ax = plt.Axes(fig, [0.1, 0.2, 0.85, 0.7])
            fig.add_axes(ax)
        plt.pause(0.1)
        
    if fig is not None and ax is None:
        fig.show()
        if fig.get_axes()!=[] and ax==None:        
            ax=fig.get_axes()[-1]
        else:
            ax = plt.Axes(fig, [0.1, 0.25, 0.85, 0.75])
    if fig is None and ax is not None:
        fig=ax.figure
#        fig.show()
#    if fig is not None and ax is not None:
##        fig.show()    
    return fig,ax
    
def opyfField(grid_val,dConfig=None,fig=None,ax=None,extent=None,extentr=None,**args):
    fig,ax=getax(fig=fig,ax=ax)
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm

    if 'vlim' in args:
        vlim=args.get('vlim',[grid_val.min(),grid_val.max()])
        if vlim is None:
            vlim=[grid_val.min(),grid_val.max()]
        del args['vlim']
    else:
        vlim=[grid_val.min(),grid_val.max()]


    args['vmin']=vlim[0]
    args['vmax']=vlim[1]
        
    im=ax.imshow(grid_val,extent=extent,**args)

    if extentr is not None: 
        ax.set_xlim(extentr[0],extentr[1])
        ax.set_ylim(extentr[2],extentr[3])
        
    return fig,ax,im  

def opyfField2(grid_x,grid_y,Field,fig=None,ax=None,**args):
#    dens = 'num de pas de res par vecteur'

    fig,ax=getax(fig=fig,ax=ax)
    if 'vlim' in args:
        vlim=args.get('vlim',[Field.min(),Field.max()])
        if vlim is None:
            vlim=[Field.min(),Field.max()]
        del args['vlim']
    else:
        vlim=[Field.min(),Field.max()]


    args['vmin']=vlim[0]
    args['vmax']=vlim[1]
    if 'cmap' not in args:
        args['cmap']= mpl.cm.coolwarm
    cmap=args.get('cmap',mpl.cm.coolwarm)
    del args['cmap']

    pc=ax.pcolormesh(grid_x,grid_y,Field,cmap=cmap,**args)

    return fig,ax,pc    
    
def opyfContour( grid_x, grid_y,grid_val,fig=None,ax=None,**args):
         
    fig,ax=getax(fig=fig,ax=ax)            
    fontsize=args.get('fontsize', None)
    DisplayNum=args.get('DisplayNum',None)
     
    cf1=ax.contour(grid_x,grid_y,grid_val,origin='upper',**args)
    cf1.levels = [nf(val) for val in cf1.levels]  
    if DisplayNum is True:
        plt.clabel(cf1, inline=2, fontsize=fontsize,fmt='%1.0f')
        
    return fig,ax,cf1    
    
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()
def CLAHEbrightness(frame,value,tileGridSize=(2,2),clipLimit=2):
#Apply the CLAHE (Enhance contrast on the frame)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize,)
    vis=np.zeros_like(frame)
    if len(np.shape(frame))==3:
        vis[:,:,0] = clahe.apply(frame[:,:,0])
        vis[:,:,1] = clahe.apply(frame[:,:,1])
        vis[:,:,2] = clahe.apply(frame[:,:,2])
    if len(np.shape(frame))==2:
        vis[:,:] = clahe.apply(frame[:,:])

                #Define the brightness            
    vis=np.where((255 - vis) < value,255,vis+value)

    return vis    
