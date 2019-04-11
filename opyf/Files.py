#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:00:40 2017

@author: Gauthier ROUSSEAU
"""
import os
import csv
import numpy as np



def mkdir2(path):     
     if not os.path.isdir(path):
         os.mkdir(path) 

def write_csvField(data,csvpath):
    h , w = data.shape
    f=open(csvpath,'w')
    writer = writer = csv.writer(f, delimiter=';')       
    for ip in range(0,h):
        writer.writerow(data[ip,:]) 
    f.close()
    
def read_csvField(csvpath):
    f=open(csvpath,'r')
    reader  = csv.reader(f, delimiter=';',quoting=csv.QUOTE_NONNUMERIC)   
    data=[]    
    for r in reader:
        data.append(r) 
    f.close()
    data=np.array(data)
    return data
    
def read_csvTrack(csvpath):
    f=open(csvpath,'r')
    reader  = csv.reader(f, csv.QUOTE_NONNUMERIC)   
    index=0
    headerTrack=[]
    datasTrack=[]
    for row in reader: 
        if index==0:
            temp=[item for number, item in enumerate(row)]
            headerTrack.append(temp)
        if index>0:
            temp=[float(item) for number, item in enumerate(row)]
            datasTrack.append(temp)
        index+=1
    headerTrack=np.array(headerTrack)
    datasTrack=np.array(datasTrack)
    return headerTrack, datasTrack
 

#def read_csv(csvpath,delimiter=','):
# 
#    f=open(csvpath,'r')
#
#    reader  = csv.reader(f,delimiter=delimiter)  
##    reader  = csv.reader(f,  dialect='excel')   
#    index=0
#    header=[]
#    datas=[]
#   
#    for row in reader: 
#        if index==0:
#            temp=[item for number, item in enumerate(row)]
#            header.append(temp)
#        if index>0:
#            temp=[float(item) for number, item in enumerate(row)]
#            datas.append(temp)
#        index+=1
#
#    header=np.array(header)
#    datas=datas[1::2]
#    datas=np.array(datas)
#    f.close()
#    return header, datas

def read_csv(csvpath,delimiter=','):
 
    f=open(csvpath,'r')
    reader  = csv.reader(f,delimiter=delimiter)  
#    reader  = csv.reader(f,  dialect='excel')   
    index=0
    header=[]
    datas=[]   
    for row in reader: 
        if index==0:
            temp=[item for number, item in enumerate(row)]
            header.append(temp)
        if index>0:
            temp=[float(item) for number, item in enumerate(row)]
            datas.append(temp)
        index+=1
        
    header=np.array(header)

    datas=np.array(datas)
    f.close()
    return header, datas


def write_csvTrack2D(csvTrackfile,X,V):
    f=open(csvTrackfile,'w')
    writer = csv.DictWriter(f, fieldnames= ['N','X','Y','Vx','Vy'])
    writer.writeheader()
    for i in range(len(X)):
        writer.writerow({'N' : i+1,'X' : X[i][0],'Y' : X[i][1],'Vx' : V[i][0],'Vy': V[i][1]}) 
    f.close()
    
    
    
    
    
def write_csvBeads(csvpath,X,R):
    f=open(csvpath,'w')
    writer = csv.DictWriter(f, fieldnames= ['N','X','Y','Z','R'])
    writer.writeheader()
    for i in range(len(X)):
        writer.writerow({'N' : i+1,'X' : X[i][0],'Y' : X[i][1],'Z' : X[i][2],'R' : R[i]}) 
    f.close()    
 
    
def write_csvTrack3D(csvTrackfile,X,V):
    f=open(csvTrackfile,'w')
    writer = csv.DictWriter(f, fieldnames= ['N','X','Y','Z','Vx','Vy'])
    writer.writeheader()
    for i in range(len(X)):
        writer.writerow({'N' : i+1,'X' : X[i][0],'Y' : X[i][1],'Z' : X[i][2],'Vx' : V[i][0],'Vy': V[i][1]}) 
    f.close() 
    
def write_csvScalar3D(filename,X,variables):

    def varline(variables, id):
        s = ""
        for v in variables:
            s = s + ',' +str(v[1][id])
        s = s + '\n'
        return s

 
    f = open(filename, "wt")
 
    f.write('X,Y,Z')

    for v in variables:
        f.write(',"%s"' % v[0])
    f.write('\n')
 

    
    id = 0

    for i in range(len(X)):
        f.write(str(X[i][0]) +','+ str(X[i][1]) +','+ str(X[i][2]))
        f.write(varline(variables, id))
        id = id + 1
 
    f.close()

def write_csvScalar1D(filename,X,variables):

    def varline(variables, id):
        s = ""
        for v in variables:
            s = s + ',' +str(v[1][id])
        s = s + '\n'
        return s

 
    f = open(filename, "wt")
 
    f.write('Z')

    for v in variables:
        f.write(',"%s"' % v[0])
    f.write('\n')
 

    
    id = 0

    for i in range(len(X)):
        f.write(str(X[i]))
        f.write(varline(variables, id))
        id = id + 1
 
    f.close()
    



def write_csvScalar(filename,variables):

    f = open(filename, "wt")
 
    i=0
    for v in variables:
        if i==0:
            f.write('"%s"' % v[0])
        else:
            f.write(',"%s"' % v[0])
        i+=1
    f.write('\n')
 

    for i in range(len(variables[0][1])):
        s = ""
        ii=0
        for v in variables:
            if ii==0:
                s = s + str(v[1][i])
            else:
                s = s + ',' +str(v[1][i])
            ii+=1
        s = s + '\n'
        
        f.write(s)

 
    f.close()    
    
    
def initializeSeqVec(seqIm_params,listD):
    prev=[]
    select=[]
    if seqIm_params['seqType']=='ABAB':
        for kl in range(0,len(listD)):
            if kl%seqIm_params['shift']==0 and kl+1<len(listD):
                select.append(kl)
                prev.append(False)
                select.append(kl+1)
                prev.append(True)            
    elif seqIm_params['seqType']=='ABCD':
        for kl in range(0,len(listD)):
            if kl==0:
                prev.append(False)
                select.append(kl)
            else:            
                select.append(kl)
                prev.append(True)
            
    return select, prev




def tecplot_WriteRectilinearMesh(filename, Xvec, Yvec, variables):
    def pad(s, width):
        s2 = s
        while len(s2) < width:
            s2 = ' ' + s2
        if s2[0] != ' ':
            s2 = ' ' + s2
        if len(s2) > width:
            s2 = s2[:width]
        return s2
    def varline(variables, id, fw):
        s = ""
        for v in variables:
            s = s + pad(str(v[1][id]),fw)
        s = s + '\n'
        return s
 
    fw = 10 # field width
 
    f = open(filename, "wt")
 
    f.write('Variables="x","y"')

    for v in variables:
        f.write(',"%s"' % v[0])
    f.write('\n\n')
 
#    f.write('Zone I=' + pad(str(len(Xvec)),6) + ',J=' + pad(str(len(Yvec)),6))

#    f.write(', F=POINT\n')
 

    
    id = 0
    for j in xrange(len(Yvec)):
        for i in xrange(len(Xvec)):
            f.write(pad(str(Xvec[i]),fw) + pad(str(Yvec[j]),fw))
            f.write(varline(variables, id, fw))
            id = id + 1
 
    f.close()
    
def tecplot_ReadRectilinearMesh(filename):
    def pad(s, width):
        s2 = s
        while len(s2) < width:
            s2 = ' ' + s2
        if s2[0] != ' ':
            s2 = ' ' + s2
        if len(s2) > width:
            s2 = s2[:width]
        return s2
    def varline(variables, id, fw):
        s = ""
        for v in variables:
            s = s + pad(str(v[1][id]),fw)
        s = s + '\n'
        return s
 
    fw = 10 # field width
 
    f = open(filename, "wt")
 
    f.write('Variables="x","y"')

    for v in variables:
        f.write(',"%s"' % v[0])
    f.write('\n\n')
 
#    f.write('Zone I=' + pad(str(len(Xvec)),6) + ',J=' + pad(str(len(Yvec)),6))

#    f.write(', F=POINT\n')
 

    
    id = 0
    for j in xrange(len(Yvec)):
        for i in xrange(len(Xvec)):
            f.write(pad(str(Xvec[i]),fw) + pad(str(Yvec[j]),fw))
            f.write(varline(variables, id, fw))
            id = id + 1
 
    f.close()  
    
def tecplot_reader(filetec,headerlines=2):
    """Tecplot reader."""
    arrays = []
    with open(filetec, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < headerlines:
                continue
            else:
                arrays.append([float(s) for s in line.split()])
    output = np.array(arrays)

    return output