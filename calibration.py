def calibration(data)
import numpy as np

#data = np.add.accumulate(np.random.random((13,3)))

data = data - data[0,:]

pts = data[(0,1,2,3),:]
pts=np.concatenate((pts.T,np.average(data[(4,9,11),:], axis = 0).reshape(3,1)), axis =1)
x,y,z = pts
# this will find the slope and x-intercept of a plane
# parallel to the y-axis that best fits the data
A_xz = np.vstack((x, np.ones(len(x)))).T
m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]

# again for a plane parallel to the x-axis
A_yz = np.vstack((y, np.ones(len(y)))).T
m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]

# the intersection of those two planes and
# the function for the line would be:
# z = m_yz * y + c_yz
# z = m_xz * x + c_xz
# or:
def liny(z):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y

zy = z[4];
xy,yy = liny(zy)

yaxis = asarray(xy,yy,zy)


pts = data[(0,5,6,7),:]
pts=np.concatenate((pts.T,np.average(data[(8,10,12),:], axis = 0).reshape(3,1)), axis =1)
x,y,z = pts 
# this will find the slope and x-intercept of a plane
# parallel to the y-axis that best fits the data
A_xz = np.vstack((x, np.ones(len(x)))).T
m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]

# again for a plane parallel to the x-axis
A_yz = np.vstack((y, np.ones(len(y)))).T
m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]

# the intersection of those two planes and
# the function for the line would be:
# z = m_yz * y + c_yz
# z = m_xz * x + c_xz
# or:
def linx(z):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y

zx = z[4];
xx,yx = linx(zx)

xaxis = asarray(xx,xy,xz)


return yaxis, xaxis
