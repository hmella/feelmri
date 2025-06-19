import numpy as np

# DENSE magnetization
# def DENSE_Mxy0(mask, M, M0, alpha, prod, t, T1, ke, X, u, phi):
#     return mask*(+0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u)
#          + 0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u))
#          + M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)*np.exp(1j*phi)

# # Complementary DENSE magnetization 
# def DENSE_Mxy1(mask, M, M0, alpha, prod, t, T1, ke, X, u, phi): 
#     return mask*(-0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u)
#         + -0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u))
#         + M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)*np.exp(1j*phi)

# # Reference DENSE magnetization
# def DENSE_Mxyin(mask, M, M0, alpha, prod, t, T1, ke, phi):
#     return mask*(M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)*np.exp(1j*phi)

# DENSE magnetization
def DENSE_Mxy0(M, M0, alpha, prod, t, T1, ke, X, u):#, phi)::
    return (+0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u)
         + 0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u))
         + M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)#*np.exp(1j*phi)

# Complementary DENSE magnetization 
def DENSE_Mxy1(M, M0, alpha, prod, t, T1, ke, X, u):#, phi): 
    return (-0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u)
        + -0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u))
        + M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)#*np.exp(1j*phi)

# Reference DENSE magnetization
def DENSE_Mxyin(M, M0, alpha, prod, t, T1, ke):#, phi):
    return (M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)#*np.exp(1j*phi)


# DENSE magnetizations
def DENSEMagnetizations():
    return DENSE_Mxy0, DENSE_Mxy1, DENSE_Mxyin
