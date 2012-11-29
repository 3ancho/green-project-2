from joy import *
from se3 import screw, unscrew, seToSE, cumdot
import sys
import ckbot.logical
import time
import math
from collections import deque
from numpy import *  
from scipy.linalg import * 
from matplotlib.pyplot import *


def jacobian_cdas( func, scl, lint=0.8, tol=1e-12, eps = 1e-30, withScl = False ):
  """Compute Jacobian of a function based on auto-scaled central differences.
  
  INPUTS:
    func -- callable -- K-vector valued function of a D-dimensional vector
    scl -- D -- vector of maximal scales allowed for central differences
    lint -- float -- linearity threshold, in range 0 to 1. 0 disables
         auto-scaling; 1 requires completely linear behavior from func
    tol -- float -- minimal step allowed
    eps -- float -- infinitesimal; must be much smaller than smallest change in
         func over a change of tol in the domain.
    withScl -- bool -- return scales together with Jacobian
  
  OUTPUTS: jacobian function 
    jFun: x --> J (for withScale=False)
    jFun: x --> J,s (for withScale=True)
    
    x -- D -- input point
    J -- K x D -- Jacobian of func at x
    s -- D -- scales at which Jacobian holds around x
  """
  scl = abs(asarray(scl).flatten())
  N = len(scl)  
  lint = abs(lint)
  def centDiffJacAutoScl( arg ):
    """
    Algorithm: use the value of the function at the center point
      to test linearity of the function. Linearity is tested by 
      taking dy+ and dy- for each dx, and ensuring that they
      satisfy lint<|dy+|/|dy-|<1/lint
    """
    x0 = asarray(arg).flatten()    
    y0 = func(x0)
    s = scl.copy()
    #print "Jac at ",x0
    idx = slice(None)
    dyp = empty((len(s),len(y0)),x0.dtype)
    dyn = empty_like(dyp)
    while True:
      #print "Jac iter ",s
      d0 = diag(s)
      dyp[idx,:] = [ func(x0+dx)-y0 for dx in d0[idx,:] ]
      dypc = dyp.conj()
      dyn[idx,:] = [ func(x0-dx)-y0 for dx in d0[idx,:] ]
      dync = dyn.conj()      
      dp = sum(dyp * dypc,axis=1)
      dn = sum(dyn * dync,axis=1)
      nul = (dp == 0) | (dn == 0)
      if any(nul):
        s[nul] *= 1.5
        continue
      rat = dp/(dn+eps)
      nl = ((rat<lint) | (rat>(1.0/lint)))
      # If no linearity violations found --> done
      if ~any(nl):
        break
      # otherwise -- decrease steps
      idx, = nl.flatten().nonzero()
      s[idx] *= 0.75
      # Don't allow steps smaller than tol
      s[idx[s[idx]<tol]] = tol
      if all(s[idx]<tol):
        break
    res = ((dyp-dyn)/(2*s[:,newaxis])).T
    if withScl:
      return res, s
    return res
  return centDiffJacAutoScl 

class Arm( object ):
  def __init__(self):
    # link lengths
    self.ll = asarray([3,4.55,8])
    # arm geometry to draw
    d=0.2
    hexa = asarray([
        [ 0, d,1-d, 1, 1-d, d, 0],
        [ 0, 1,  1, 0,  -1,-1, 0],
        [ 0, 0,  0, 0,   0, 0, 0],
        [ 1, 1,  1, 1,   1, 1, 1],
    ]).T
    sqr = asarray([
        [ d, d, d, d, d, 1-d, 1-d, 1-d, 1-d, 1-d],
        [ 1, 0,-1, 0, 1, 1, 0,-1, 0, 1 ],
        [ 0, 1, 0,-1, 0, 0, 1, 0,-1, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    ]).T

    geom = concatenate([
      hexa, hexa[:,[0,2,1,3]], sqr,
    ], axis=0)

    self.geom = [( asarray([[0,0,0,1]]) ).T ]
    tw = []
    LL = 0

    for n,ll in enumerate(self.ll):
      self.geom.append( 
        ( asarray([ll,1,2,1])*geom+[LL,0,0,0] ).T
      )
      if n == 0:
        w = asarray([0,0,1])
      else:
        w = asarray([0,1,0])
      print n, " and ", w

      v = -cross(w,[LL,0,0])
      tw.append( concatenate([v,w],0) )
      LL += ll

    self.tw = asarray(tw)
    self.tool = asarray([LL,0,0,1]).T
    # overwrite method with jacobian function
    self.getToolJac = jacobian_cdas( 
      self.getTool, ones(self.tw.shape[0])*0.05 
    )
  
  def at( self, ang ):
    """
    Compute the rigid transformations for a 3 segment arm
    at the specified angles
    """
    ang = asarray(ang)[:,newaxis]
    tw = ang * self.tw
    A = [identity(4)]
    for twi in tw:
      M = seToSE(twi)
      A.append(dot(A[-1],M))
    return A
    
  def getTool( self, ang ):
    """
    Get "tool tip" position in world coordinates
    """
    M = self.at(ang)[-1]
    return dot(M, self.tool)
  
  def getToolJac( self, ang ):
    """
    Get "tool tip" Jacobian by numerical approximation
    
    NOTE: implementation is a placeholder. This method is overwritten
    dynamically by __init__() to point to a jacobian_cdas() function
    """
    raise RuntimeError("uninitialized method called")
    
  def plotIJ( self, ang, axI=0, axJ=1 ):
    """
    Display the specified axes of the arm at the specified set of angles
    """
    A = self.at(ang)
    for a,g in zip(A, self.geom):
      ng = dot(a,g)
      plot( ng[axI,:], ng[axJ,:], '.-' )
    tp = dot(a, self.tool)
    plot( tp[axI], tp[axJ], 'hk' )
    plot( tp[axI], tp[axJ], '.y' )
    

  def plot3D( self, ang ):
    ax = [-20,20,-20,20]
    subplot(2,2,1)
    self.plotIJ(ang,0,1)
    axis('equal')
    axis(ax)
    grid(1)
    xlabel('X'); ylabel('Y')

    subplot(2,2,2)
    self.plotIJ(ang,1,2)
    axis('equal')
    axis(ax)
    grid(1)
    xlabel('Y'); ylabel('Z')

    subplot(2,2,3)
    self.plotIJ(ang,0,2)
    axis('equal')
    axis(ax)
    grid(1)
    xlabel('X'); ylabel('Z')
  
class ArmApp( JoyApp ):
  def __init__(self, factor = 0.1, testing=False, *arg, **kw):
    JoyApp.__init__(self, robot = {'count': 3},  *arg,**kw)
    # short-cut for modules
    self.top = self.robot.at.top
    self.mid = self.robot.at.mid
    self.bot = self.robot.at.bot
    self.factor = factor
    self.testing = testing

  def _get_ang(self):
    poses = [self.bot.get_pos(), self.mid.get_pos(), self.top.get_pos()] 
    ang = [item * pi / 18000 for item in poses]
    return ang

   
  def onStart(self):
    self.arm = Arm()
    self.ang = [0,0,0]
    self.top.mem[self.top.mcu.torque_limit] = 200
    self.mid.mem[self.mid.mcu.torque_limit] = 200
    self.bot.mem[self.bot.mcu.torque_limit] = 200
    if self.testing:
      self.simulator_plan = SimulatorPlan(self) 
      self.simulator_plan.start()
      progress("simulator started") 

  def onEvent(self,evt):

    # Exit
    if evt.type == KEYDOWN and evt.key in [ K_ESCAPE ]: # Esc 
        progress("Exiting!")
        self.stop()

    if evt.type == KEYDOWN and evt.key == K_s: # go slack 
      self.top.go_slack()
      self.mid.go_slack()
      self.bot.go_slack()
      self.simulator_plan.stop()
  
    if evt.type == KEYDOWN and evt.key == K_m: # Move a vector 
      d = input("input a direction as list > ")
      self.move(d) 

    if evt.type == KEYDOWN and evt.key == K_r: #
      progress( "Reset to origin pos")
      self.top.set_pos(0)
      self.mid.set_pos(0)
      self.bot.set_pos(0)

    if evt.type == KEYDOWN and evt.key == K_p: # Print real world coordinates
      ang = self._get_ang() 
      # draw
      self.simulator_plan.simulator_draw(ang)

      print self.arm.getTool(ang) 

  def onStop(self):
    for i in range(3):
        self.robot.off()
    progress("The application have been stopped.")
    return super( ArmApp, self).onStop()

  def move(self, d): # d is a vector
    self.top.mem[self.top.mcu.torque_limit] = 200
    self.mid.mem[self.mid.mcu.torque_limit] = 200
    self.bot.mem[self.bot.mcu.torque_limit] = 200
    
    if type(d) != list or len(d) != 3:
      progress("input format error")
      return

    pre_ang = self.ang
    Jt = self.arm.getToolJac(self.ang)
    self.ang = self.ang + dot(pinv(Jt)[:,:len(d)],d)
    print "pre angles: ", pre_ang
    print "Angles: ", self.ang
    for i in range(3):
      if abs(self.ang[i] - pre_ang[i]) > pi/2:
        print "angle range error"
        return

    length = math.sqrt( d[0]**2 + d[1]**2 + d[2]**2 ) 
    n = int( length / self.factor) + 1

    pos = asarray([self.top.get_pos(), self.mid.get_pos(), self.bot.get_pos()])
    step = asarray(self.ang) * 18000 / math.pi / n # [a,b,c]
    
    for i in range(n):
      time.sleep(0.2)
      print pos
      pos += step
      self.top.set_pos(pos[0])
      self.mid.set_pos(pos[1])
      self.bot.set_pos(pos[2])

class SimulatorPlan( Plan ):
  def __init__(self, app, intervel=0.8):
    Plan.__init__(self, app)
    self.intervel = intervel

  def onStart(self):
    self.f = gcf()

  def simulator_draw(self, ang=None):
    if not ang:
      ang = self.app._get_ang() 
    if ang:  
      print ang
      # draw
      self.f.set(visible=0)
      clf()
      self.app.arm.plot3D(ang)
      self.f.set(visible=1)
      draw()

  def behavior(self):
    while True:
      self.simulator_draw()
      yield self.forDuration(self.intervel)

def main():
    app = ArmApp(testing=True)
    app.run()

if __name__ == '__main__':
    main()
