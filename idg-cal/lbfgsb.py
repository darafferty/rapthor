import numpy as np
import math

class LBFGSB():
    """Implements LBFGS-B algorithm (with box-constraints).
     Primary reference: 
      1) MATLAB code https://github.com/bgranzow/L-BFGS-B by Brian Granzow
     Theory based on:
      1) A Limited Memory Algorithm for Bound Constrained Optimization, Byrd et al. 1995
      2) Numerical Optimization, Nocedal and Wright, 2006

      Example:
      N=2
      x0=np.zeros((N,1))
      x0[0]=-1.2
      x0[1]=1.0
      x_l=np.ones((N,1))*(-1.0)
      x_u=np.ones((N,1))*(1.0)
      x_l[0]=-1
      x_l[1]=0.5
      x_u[0]=0.5
      x_u[1]=1.5
      optimizer=LBFGSB(x0,x_l,x_u,max_iter=30)

      def gradient(x):
        x=x.squeeze()
        g=np.zeros((N,1))
        g[0]=2.0*(200.0*x[0]*x[0]*x[0]-200.0*x[0]*x[1]+x[0]-1.0)
        g[1]=200.0*(x[1]-x[0]*x[0])
        return g
      def cost(x):
        x=x.squeeze()
        f=math.pow(1.0-x[0],2.0)+100.0*math.pow(x[1]-x[0]*x[0],2.0)
        return f

      f=optimizer.step(cost,gradient)
      print(f'cost {f} solution {optimizer.x.squeeze()}')

    """
    def __init__(self,x0,lower_bound,upper_bound,max_iter=2,history_size=7):
        self.n=x0.size
        assert(self.n==lower_bound.size)
        assert(self.n==upper_bound.size)
        self.m=history_size
        self.max_iter=max_iter
        self.theta=1.0
        self.x=x0.copy()
        self.g=None
        self.l=lower_bound
        self.u=upper_bound
        self.tol=1e-9
        self.W=np.zeros((self.n,self.m*2))
        self.Y=np.zeros((self.n,self.m))
        self.S=np.zeros((self.n,self.m))
        self.M=np.zeros((self.m*2,self.m*2))

        # float constants
        self.realmax=1e+20
        self.eps=1e-20

        self.fit_to_constraints_(self.x,self.l,self.u)


    def grad_check(self,cost,gradient):
        x=np.random.rand(self.n,1)
        g0=gradient(x)
        g=np.zeros((self.n,1))
        eps=1e-6
        for ci in range(self.n):
            x[ci,0]+=eps
            f0=cost(x)
            x[ci,0]-=2.0*eps
            f1=cost(x)
            g[ci,0]=(f0-f1)/(2.0*eps)
            x[ci,0]+=eps
            print(f'{ci} cost {f0} {f1} g0/g={g0[ci,0]/g[ci,0]}')

        print(np.linalg.norm(g0-g)/np.linalg.norm(g))

    def step(self,cost,gradient):
        # cost f(x) R^n->1
        # gradient g(x) R^n->R^n
        f=cost(self.x)
        g=gradient(self.x)
        n_iter=0
        while (self.get_optimality_(self.x,g,self.l,self.u)> self.tol) and (n_iter < self.max_iter):
            x_old=self.x.copy()
            g_old=g.copy()
            # compute new search direction
            xc,c=self.get_cauchy_point_(self.x,g,self.l,self.u,self.theta,self.W,self.M)
            xbar,line_search_flag=self.subspace_min_(self.x,g,self.l,self.u,xc,c,self.theta,self.W,self.M)
            alpha=1.0
            if (line_search_flag):
                alpha=self.strong_wolfe_(cost,gradient,self.x,f,g,xbar-self.x)
            self.x = self.x+ alpha*(xbar-self.x)


            f=cost(self.x)
            g=gradient(self.x)
            y=g-g_old
            s=self.x-x_old

            print(f'iter {n_iter} cost {f} grad norm {np.linalg.norm(g)}')

            curv=(s.transpose() @ y)
            if (curv<self.eps):
                print('Warning: negative curvature detected, skipping update')
                n_iter+=1
                continue
            if (n_iter<self.m):
                self.Y[:,n_iter]=y.squeeze()
                self.S[:,n_iter]=s.squeeze()
            else:
                self.Y[:,0:self.m-1]=self.Y[:,1:self.m]
                self.S[:,0:self.m-1]=self.S[:,1:self.m]
                self.Y[:,-1] = y.squeeze()
                self.S[:,-1] = s.squeeze()

            self.theta=(y.transpose() @ y)/(y.transpose() @ s)
            self.W[:,0:self.m]=self.Y
            self.W[:,self.m:2*self.m]=self.theta*self.S
            A = self.S.transpose() @ self.Y
            L=np.tril(A,-1)
            D=-1.0*np.diag(np.diag(A))
            MM=np.zeros((2*self.m,2*self.m))
            MM[0:self.m,0:self.m]=D
            MM[0:self.m,self.m:2*self.m]=L.transpose()
            MM[self.m:2*self.m,0:self.m]=L
            MM[self.m:2*self.m,self.m:2*self.m]=self.theta*self.S.transpose() @ self.S
            self.M=np.linalg.pinv(MM)
                

            n_iter+=1
        
        # check why we stopped
        if (n_iter==self.max_iter):
            print('Reached maximum number of iterations, stopping')
        if (self.get_optimality_(self.x,g,self.l,self.u) <= self.tol):
            print('Reached required convergence tolerance, stopping')

        # return the result dict
        return {'residual': f, 'success': False, 'x': self.x}


    def strong_wolfe_(self,cost,gradient,x0,f0,g0,p):
        # line search to satisfy strong Wolfe conditions
        # Alg 3.5, pp. 60, Numerical optimization Nocedal & Wright
        # cost: cost function R^n -> 1
        # gradient: gradient function R^n -> R^n
        # x0: nx1 initial parameters
        # f0: 1 intial cost
        # g0: nx1 initial gradient
        # p: nx1 intial search direction
        # out:
        # alpha: step length

        c1=1e-4
        c2=0.9
        alpha_max=2.5
        alpha_im1=0
        alpha_i =1
        f_im1=f0
        dphi0=g0.transpose() @ p
        i=0
        max_iters=20
        while 1:
            x=x0+alpha_i*p
            f_i=cost(x)
            if (f_i>f0+c1*dphi0) or ((i>1) and (f_i>f_im1)):
                alpha=self.alpha_zoom_(cost,gradient,x0,f0,g0,p,alpha_im1,alpha_i)
                break
            g_i=gradient(x)
            dphi=g_i.transpose() @ p
            if (abs(dphi)<= -c2*dphi0):
                alpha=alpha_i
                break
            if (dphi>=0.0):
                alpha=self.alpha_zoom_(cost,gradient,x0,f0,g0,p,alpha_i,alpha_im1)
                break
            alpha_im1=alpha_i
            f_im1=f_i
            alpha_i=alpha_i + 0.8*(alpha_max-alpha_i)
            if (i>max_iters):
                alpha=alpha_i
                break
            i=i+1

        return alpha

    def alpha_zoom_(self,cost,gradient,x0,f0,g0,p,alpha_lo,alpha_hi):
        # Alg 3.6, pp. 61, Numerical optimization Nocedal & Wright
        # cost: cost function R^n -> 1
        # gradient: gradient function R^n -> R^n
        # x0: nx1 initial parameters
        # f0: 1 intial cost
        # g0: nx1 initial gradient
        # p: nx1 intial search direction
        # alpha_lo: low limit for alpha
        # alpha_hi: high limit for alpha
        # out:
        # alpha: zoomed step length
        c1=1e-4
        c2=0.9
        i=0
        max_iters=20
        dphi0=g0.transpose() @ p

        while 1:
            alpha_i=0.5*(alpha_lo+alpha_hi)
            alpha=alpha_i
            x=x0+alpha_i*p
            f_i=cost(x)
            x_lo=x0+alpha_lo*p
            f_lo=cost(x_lo)
            if ((f_i>f0+c1*alpha_i*dphi0) or (f_i>=f_lo)):
                alpha_hi=alpha_i
            else:
                g_i=gradient(x)
                dphi=g_i.transpose() @ p
                if ((abs(dphi)<= -c2*dphi0)):
                    alpha=alpha_i
                    break
                if (dphi*(alpha_hi-alpha_lo)>=0.0):
                    alpha_hi=alpha_lo

                alpha_lo=alpha_i
            i=i+1
            if (i>max_iters):
                alpha=alpha_i
                break

        return alpha


    def fit_to_constraints_(self,x,l,u):
        # fit x to fit constraints
        for i in range(self.n):
            if x[i]<l[i]:
                x[i]=l[i]
            elif x[i]>u[i]:
                x[i]=u[i]

    def get_optimality_(self,x,g,l,u):
        # get the inf-norm of the projected gradient
        # pp. 17, (6.1)
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        projected_g=x-g
        for i in range(self.n):
            if (projected_g[i]<l[i]):
                projected_g[i]=l[i]
            elif (projected_g[i]>u[i]):
                projected_g[i]=u[i]

        projected_g=projected_g-x
        return max(abs(projected_g))

    def get_breakpoints_(self,x,g,l,u):
        # compute breakpoints for Cauchy point
        # pp 5-6, (4.1), (4.2), pp. 8, CP initialize \mathcal{F}
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # out:
        # t: nx1 breakpoint vector
        # d: nx1 search direction vector
        # F: nx1 indices that sort t from low to high
        t=np.zeros((self.n,1))
        d=-g
        for i in range(self.n):
            if (g[i]<0.0):
                t[i]=(x[i]-u[i])/g[i]
            elif (g[i]>0.0):
                t[i]=(x[i]-l[i])/g[i]
            else:
                t[i]=self.realmax
            if (t[i]<self.eps):
                d[i]=0.0

        F=np.argsort(t.squeeze())

        return t,d,F


    def get_cauchy_point_(self,x,g,l,u,theta,W,M):
        # Generalized Cauchy point
        # pp. 8-9, algorithm CP
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # theta: >0, scaling
        # W: nx2m 
        # M: 2mx2m
        # out:
        # xc: nx1 the generalized Cauchy point
        # c: 2mx1 initialization vector for subspace minimization

        tt,d,F=self.get_breakpoints_(x,g,l,u)
        xc=x.copy()
        p=W.transpose() @ d # W^T  d
        c=np.zeros((2*self.m,1))
        fp=-d.transpose() @ d # f' = - d^T  d
        fpp=-theta*fp-p.transpose() @ M @ p # -theta f' - p^T M p
        # shed array dims
        fp=fp.squeeze()
        fpp=fpp.squeeze()

        fpp0=-theta*fp
        if fpp != 0.0:
          dt_min=-fp/fpp
        else:
          dt_min=-fp/self.eps
        t_old=0
        for j in range(self.n):
            i=j
            if (F[i]>=0): # index 0 can also work
              break
        b=F[i]    
        t=tt[b]
        dt=t-t_old

        while (i<self.n) and (dt_min>dt):
            if d[b]>0.0:
                xc[b]=u[b]
            elif d[b]<0.0:
                xc[b]=l[b]

            zb=xc[b]-x[b]
            c=c+dt*p.reshape(c.shape)
            gb=g[b]
            Wbt=W[b,:]
            fp=fp+dt*fpp+gb*gb+theta*gb*zb-gb*(Wbt @ M @ c)
            fpp=fpp-theta*gb*gb-2.0*gb*(Wbt @ M @ p) - gb*gb*(Wbt @ M @ Wbt.transpose())
            # shed array dims
            fp=fp.squeeze()
            fpp=fpp.squeeze()
            fpp=max(self.eps*fpp0,fpp)
            p=p+gb*Wbt.reshape(p.shape) # no need for Wbt.transpose()
            d[b]=0.0
            if fpp != 0.0:
              dt_min=-fp/fpp
            else:
              dt_min=-fp/self.eps
            t_old=t
            i=i+1
            if (i<self.n):
                b=F[i]
                t=tt[b]
                dt=t-t_old


        dt_min=max(dt_min,0)
        t_old=t_old+dt_min
        for j in range(i,self.n):
            idx=F[j]
            xc[idx]=x[idx]+t_old*d[idx]

        c=c+dt_min*p.reshape(c.shape)


        return xc,c

    def subspace_min_(self,x,g,l,u,xc,c,theta,W,M):
        # subspace minimization for the quadratic model over free variables
        # direct primal method, pp 12
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # xc: nx1 generalized Cauchy point
        # c: 2mx 1 minimization initialization vector
        # theta: >0, scaling
        # W: nx2m 
        # M: 2mx2m
        # out:
        # xbar: nx1 minimizer 
        # line_search_flag: bool

        line_search_flag=True
        # build a list of free variables
        free_vars_index=list()
        for i in range(self.n):
            if (xc[i] != u[i]) and (xc[i] != l[i]):
                free_vars_index.append(i)

        n_free_vars=len(free_vars_index)
        if n_free_vars==0:
            xbar=xc.copy()
            line_search_flag=False
            return xbar,line_search_flag

        # W^T Z : the restriction of W to free variables
        WtZ=np.zeros((2*self.m,n_free_vars))
        # each column of WtZ (2*m values) = row of i-th free variable in W (2*m values)
        for i in range(n_free_vars):
            WtZ[:,i]=W[free_vars_index[i],:]

        # reduced gradient of quadratic restricted to free variables
        rr=g+ theta*(xc-x) - W @ (M @ c)
        r=np.zeros((n_free_vars,1))
        for i in range(n_free_vars):
            r[i]=rr[free_vars_index[i]]

        # form intermediate variables
        invtheta=1.0/theta
        v = M @ WtZ @ r
        N = invtheta * WtZ @ WtZ.transpose()
        N = np.eye(2*self.m) - M @ N

        v,_,_,_= np.linalg.lstsq(N,v,rcond=-1)

        du= -invtheta*r - invtheta*invtheta * WtZ.transpose() @ v

        alpha_star = self.find_alpha_(l,u,xc,du,free_vars_index)
        d_star = alpha_star*du
        xbar = xc.copy()
        for i in range(n_free_vars):
            idx=free_vars_index[i]
            xbar[idx]=xbar[idx]+d_star[i]

        return xbar,line_search_flag



    def find_alpha_(self,l,u,xc,du,free_vars_index):
        # pp. 11, (5.8) 
        # l: nx1 lower bound
        # u: nx1 upper bound
        # xc: nx1 generalized Cauchy point
        # du: n_free_varsx1 
        # free_vars_index:  n_free_varsx1 indices of free variables
        # out:
        # alpha_star: positive scaling parameter
        n_free_vars=len(free_vars_index)
        alpha_star=1
        for i in range(n_free_vars):
            idx=free_vars_index[i]
            if (du[i]>0.0):
                alpha_star=min(alpha_star,(u[idx]-xc[idx])/du[i])
            elif (du[i]<0.0):
                alpha_star=min(alpha_star,(l[idx]-xc[idx])/du[i])
            else: # when du ~= 0
                alpha_star=alpha_star

        return alpha_star
