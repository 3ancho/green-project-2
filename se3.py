# This library is provided under the GNU LGPL v3.0 license, with the 
# additional provision that any use in work leading to academic 
# publication requires at least one of the following conditions be met:
# (1) You credit all authors of the library, by name, in the publication
# (2) You reference at least one relevant paper by each of the authors
# 
# (C) Shai Revzen, U Penn. 2011

"""
File se3.py contains a library of functions for performing operations
in the classical Lie Group SE(3) and associated algebra se(3).
"""

from numpy import zeros, zeros_like, asarray, array, allclose, resize, ones, ones_like, empty, identity, prod, sqrt, isnan, pi, cos, sin, newaxis, diag, sum, arange, dot, cross

from scipy.linalg import expm as expM

def skew( v ):
  """
  Convert a 3-vector to a skew matrix such that 
    dot(skew(x),y) = cross(x,y)
  
  The function is vectorized, such that:
  INPUT:
    v -- N... x 3 -- input vectors
  OUTPUT:
    N... x 3 x 3  
    
  For example:
  >>> skew([[1,2,3],[0,0,1]])
  array([[[ 0,  3, -2],
        [-3,  0,  1],
        [ 2, -1,  0]],
  <BLANKLINE>
       [[ 0,  1,  0],
        [-1,  0,  0],
        [ 0,  0,  0]]])
  """
  v = asarray(v).T
  z = zeros_like(v[0,...])
  return array([
      [ z, v[2,...], v[1,...]],
      [-v[2,...], z, -v[0,...] ],
      [-v[1,...], v[0,...], z ] ]).T

def unskew( S ):
  """
  Convert a skew matrix to a 3-vector
  
  The function is vectorized, such that:
  INPUT:
    S -- 3 x 3 x N... -- input skews
  OUTPUT:
    3 x N...  
  
  This is the "safe" function -- it tests for skewness first.
  Use unskew_UNSAFE(S) to skip this check
  
  Example:
  >>> x = array(range(24)).reshape(2,1,4,3); allclose(unskew(skew(x)),x)
  True
  >>> unskew([[1,2,3],[4,5,6],[7,8,9]])
  Traceback (most recent call last):
  ...
  AssertionError: S is skew
  """
  S = asarray(S)
  assert allclose(S.T.transpose((1,0)+tuple(range(2,S.ndim))),-S.T),"S is skew"
  return unskew_UNSAFE(S)

def unskew_UNSAFE(S):
  """
  Convert a skew matrix to a 3-vector
  
  The function is vectorized, such that:
  INPUT:
    S -- N... x 3 x 3 -- input skews
  OUTPUT:
    N... x 3  
  
  This is the "unsafe" function -- it does not test for skewness first.
  Use unskew(S) under normal circumstances
  """
  S = asarray(S).T
  return array([S[2,1,...],S[0,2,...],S[0,1,...]]).T

def screw( v ):
  """
  Convert a 6-vector to a screw matrix 
  
  The function is vectorized, such that:
  INPUT:
    v -- N... x 6 -- input vectors
  OUTPUT:
    N... x 4 x 4  
  """
  v = asarray(v)
  z = zeros_like(v[0,...])
  return array([
      [ z, -v[...,5], v[...,4], v[...,0] ],
      [ v[...,5],  z,-v[...,3], v[...,1] ],
      [-v[...,4],  v[...,3], z, v[...,2] ],
      [ z,         z,        z, z] ])

def unscrew( S ):
  """
  Convert a screw matrix to a 6-vector
  
  The function is vectorized, such that:
  INPUT:
    S -- N... x 4 x 4 -- input screws
  OUTPUT:
    N... x 6
  
  This is the "safe" function -- it tests for screwness first.
  Use unscrew_UNSAFE(S) to skip this check
  """
  S = asarray(S)
  assert allclose(S[...,:3,:3].transpose(0,1),-S[...,:3,:3]),"S[...,:3,:3] is skew"
  assert allclose(S[...,3,:],0),"Bottom row is 0"
  return unscrew_UNSAFE(S)

def unscrew_UNSAFE(S):
  """
  Convert a screw matrix to a 6-vector
  
  The function is vectorized, such that:
  INPUT:
    S -- N... x 4 x 4 -- input screws
  OUTPUT:
    N... x 6 
  
  This is the "unsafe" function -- it does not test for screwness first.
  Use unscrew(S) under normal circumstances
  """
  S = asarray(S)
  return array([S[...,0,3],S[...,1,3],S[...,2,3],
      S[...,1,2],S[...,2,0],S[...,0,1]])

def soToSO( aa ):
  """
  Exponential of a rotation vector, a.k.a. Rodrigues' Formula
  INPUT:
    aa -- N... x 3
  OUTPUT:
    N... x 3 x 3  
  
  >>> diag(soToSO([3.1415926,0,0])).round(2)
  array([ 1., -1., -1.])
  >>> soToSO([0,3.1415926/4,0]).round(2)
  array([[ 0.71,  0.  , -0.71],
         [ 0.  ,  1.  ,  0.  ],
         [ 0.71,  0.  ,  0.71]])
  """
  aa = asarray(aa,dtype=float)
  if aa.shape[-1] is not 3:
    raise ValueError("last dimension must be 3 but shape is %s" % str(aa.shape))    
  t = sqrt(sum(aa * aa,-1))
  k = aa / t[...,newaxis]
  k[isnan(k)]=0
  kkt = k[...,:,newaxis] * k[...,newaxis,:]
  I = identity(3)
  # Note: (a.T+b.T).T is not a+b -- index broadcasting is different
  R = (sin(t).T*skew(k).T + (cos(t)-1).T*(I-kkt).T).T + I
  return R

def seToSE( x ):
  """
  Convert a twist (a rigid velocity, element of se(3)) to a rigid
  motion (an element of SE(3))
  
  INPUT:
    x -- 6 sequence
  OUTPUT:
    result -- 4 x 4  

  """
  x = asarray(x,dtype=float)
  if x.shape != (6,):
    raise ValueError("shape must be (6,); got %s" % str(x.shape))
  #
  return expM(screw(x))

if 0: # create perturbed grid
  X,Y = meshgrid(arange(-2,3), arange(-2,3))
  X = X + randn(*X.shape)/10
  Y = Y + randn(*Y.shape)/10
  p = c_[X.flatten(), Y.flatten(), zeros(Y.size), ones(Y.size)].T

if 0:
  q = randn(2)
  tw = asarray(list(-cross([0,0,1],[q[0],q[1],0]))+[0,0,1])
  scm = screw(tw)
if 0:
  figure(2)
  clf()
  for th in arange(0,0.2,0.05):
    M1 = expM(scm[:3,:3]*th)
    p1 = dot(M1,p[:3,:])
    plot( p1[0,:], p1[1,:], 'or')
    M2 = soToSO(tw[3:]*th)
    p2 = dot(M2,p[:3,:])
    plot( p2[0,:], p2[1,:], '.b')
  plot(q[0],q[1],'dk')
  axis('equal')
  grid(1)
  
if 0:
  figure(3)
  clf()
  for th in arange(0,7,0.1):
    M1 = expM(scm*th)
    p1 = dot(M1,p)
    plot( p1[0,:], p1[1,:], 'or')
    M2 = seToSE(tw*th)
    p2 = dot(M2,p)
    plot( p2[0,:], p2[1,:], '.b')
  plot(q[0],q[1],'dk')
  axis('equal')
  grid(1)

def gendot( seq ):
  """Generator for matrix products of a sequence of matrices"""
  A = 1
  for S in seq:
    A = dot(A,S)
    yield A

def cumdot( seq ):
  """Cumulative matrix products of a sequence of matrices"""
  return list(gendot(seq))
  
