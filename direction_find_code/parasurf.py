# https://gist.github.com/subnivean/4255343
import numpy as np
from scipy.interpolate import RectBivariateSpline, bisplev


class ParaSurf(object):
    def __init__(self, u, v, xyz, bbox=[-0.25, 1.25, -0.5, 1.5], ku=3, kv=3):
        """Parametric (u,v) surface approximation over a rectangular mesh.

        Parameters
        ----------
        u, v : array_like
            1-D arrays of coordinates in strictly ascending order.
        xyz : array_like
            3-D array of (x, y, z) data with shape (3, u.size, v.size).
        bbox : array_like, optional
            Sequence of length 4 specifying the boundary of the rectangular
            approximation domain. See scipy.interpolate.RectBivariateSpline
            for more info
        ku, kv : ints, optional
            Degrees of the bivariate spline. Default is 3.
        """
        self._create_srf(u, v, xyz, bbox, ku, kv)

        self.bbox = bbox
        self.u = u
        self.v = v
        self.ku = ku
        self.kv = kv

    def _create_srf(self, u, v, xyz, bbox, ku, kv):
        # Create surface definitions
        xsrf = RectBivariateSpline(u, v, xyz[0], bbox=bbox, kx=ku, ky=kv, s=0)
        ysrf = RectBivariateSpline(u, v, xyz[1], bbox=bbox, kx=ku, ky=kv, s=0)
        zsrf = RectBivariateSpline(u, v, xyz[2], bbox=bbox, kx=ku, ky=kv, s=0)

        # Create the 5-element list suitable for feeding
        # to bisplev() later (we need bisplev() to get partial
        # derivatives, for surface normals and surface tangents
        # along u and v)
        self._xsrfdef = xsrf.tck + xsrf.degrees
        self._ysrfdef = ysrf.tck + ysrf.degrees
        self._zsrfdef = zsrf.tck + zsrf.degrees

        self._xsrf = xsrf
        self._ysrf = ysrf
        self._zsrf = zsrf
        self._origxyzs = xyz

    def ev(self, u, v, mesh=True):
        """Get point(s) on surface at (u, v).

        Parameters
        ----------
        u, v: array-like or scalar
            u and v may be scalar or vector

        mesh : boolean
            If True, will expand the u and v values into a mesh.
            For example, with u = [0, 1] and v = [0, 1]: if 'mesh'
            is True, the surface will be evaluated at [0, 0], [0, 1],
            [1, 0] and [1, 1], while if it is False, the evalation
            will only be made at [0, 0] and [1, 1]

        Returns
        -------
        If scalar values are passed for *both* u and v, returns
        a 1-D 3-element array [x,y,z]. Otherwise, returns an array
        of shape 3 x len(u) x len(v), suitable for feeding to Mayavi's
        mlab.mesh() plotting function (as mlab.mesh(*arr)).
        """
        u = np.array([u]).reshape(-1,)
        v = np.array([v]).reshape(-1,)
        if mesh:
            # I'm still not sure why we're required to flip u and v
            # below, but trust me, it doesn't work otherwise.
            V, U = np.meshgrid(v, u)
            U = U.ravel()
            V = V.ravel()
        else:
            if len(u) != len(v): # *Need* to mesh this, like above!
                V, U = np.meshgrid(v, u)
                U = U.ravel()
                V = V.ravel()
            else:
                U, V = u, v
        x = self._xsrf.ev(U, V)
        y = self._ysrf.ev(U, V)
        z = self._zsrf.ev(U, V)

        if u.shape == (1,) and v.shape == (1,):
            # Scalar u and v values; return 1-D 3-element array.
            return np.array([x, y, z]).ravel()
        else:
            # u and/or v passed as lists; return 3 x m x n array,
            # where m is len(u) and n is len(v). This format
            # is compatible with enthought.mayavi's mlab.mesh()
            # function.
#             print x.shape, y.shape, z.shape
#             print x
#             print y
#             print z
            return np.array([x, y, z]).reshape(3, len(u), -1)

    def utan(self, u, v):
        dxdu = bisplev(u, v, self._xsrfdef, dx=1, dy=0)
        dydu = bisplev(u, v, self._ysrfdef, dx=1, dy=0)
        dzdu = bisplev(u, v, self._zsrfdef, dx=1, dy=0)
        du = np.array([dxdu, dydu, dzdu])
        du /= np.sqrt((du**2).sum(axis=0))
        return du

    def vtan(self, u, v):
        dxdv = bisplev(u, v, self._xsrfdef, dx=0, dy=1)
        dydv = bisplev(u, v, self._ysrfdef, dx=0, dy=1)
        dzdv = bisplev(u, v, self._zsrfdef, dx=0, dy=1)
        dv = np.array([dxdv, dydv, dzdv])
        dv /= np.sqrt((dv**2).sum(axis=0))
        return dv

    def normal(self, u, v):
        """Get normal(s) at (u, v).
        """
        u = np.array([u]).reshape(-1,)
        v = np.array([v]).reshape(-1,)

        dxdus = bisplev(u, v, self._xsrfdef, dx=1)
        dydus = bisplev(u, v, self._ysrfdef, dx=1)
        dzdus = bisplev(u, v, self._zsrfdef, dx=1)
        dxdvs = bisplev(u, v, self._xsrfdef, dy=1)
        dydvs = bisplev(u, v, self._ysrfdef, dy=1)
        dzdvs = bisplev(u, v, self._zsrfdef, dy=1)

        if u.shape == (1,) and v.shape == (1,):
            # Scalar u and v values; return 1-D 3-element array.
            normal = np.cross([dxdus, dydus, dzdus], [dxdvs, dydvs, dzdvs])
            normal /= np.sqrt(np.dot(normal, normal))
            return normal
        else:
            # u and/or v passed as lists; return 3 x m x n array,
            # where m is len(u) and n is len(v). This format
            # is compatible with enthought.mayavi's mlab.mesh()
            # function.
            normals = np.cross([dxdus, dydus, dzdus], [dxdvs, dydvs, dzdvs],
                                axisa=0, axisb=0)
            normals /= np.sqrt((normals**2).sum(axis=2))[:, :, np.newaxis]
            return normals.T

    @property
    def xyzs(self):
        return self._origxyzs

    def offset(self, offsetamt):
        """Offset the original surface by the given amount
        (but what direction?)
        """
        normals = self.normal(self.u, self.v)
        offpts = self._origxyzs + offsetamt * normals
        return self.__class__(self.u, self.v, offpts,
                              self.bbox, self.ku, self.kv)

#     def plane_intersect_pts(self, planedef):
#         """Given a plane definition (see below), returns the set of
#         intersection points, one for each u-isocurve.
# 
#         Parameters
#         ----------
#         planedef : array_like
#             Normalized [A, B, C, D] values from the plane equation
#             `Ax + By + Cz - D = 0`.
# 
#             As a reminder, A, B and C are the (normalized) values
#             of the plane normal vector, and D is the dot product
#             of *any* origin-to-plane vector and the normalized
#             plane normal vector.
# 
#         Returns
#         -------
#         pts: Array of intersection points
#         """
# 
#         # Strategy: rotate the bispline coefficients and the
#         # original xyz values to put them into 'cut plane coords'.
#         # By doing this, we can restrict our evals and optimizations
#         # to the z surface only, which speeds things up by a factor
#         # of ~3.
#         import msk.xy_rots_from_vector as mxyr
#         zht = planedef[3]
#         tr = mxyr.trsf_from_zvec(planedef[0:3])
#         rmat = tr.mat[0:3, 0:3]
#         coeffs = np.array([
#                     self._xsrf.get_coeffs(),
#                     self._ysrf.get_coeffs(),
#                     self._zsrf.get_coeffs()])
#         rotcoeffs = np.dot(rmat.T, coeffs)
#         xyzsshape = self.xyzs.shape
#         rotxyzs = np.dot(rmat.T, self.xyzs.reshape(3,-1)).reshape(xyzsshape)
#         # Redefine (temporarily) the z-surface coefficients; we'll
#         # reset them below when we're done.
#         self._zsrf.tck = (self._zsrf.tck[0], self._zsrf.tck[1], rotcoeffs[2])
# 
#         import scipy.optimize as spo
#         def _get_intersection_v(u, vi=0.0, vf=1.0):
#             def intersectfunc(u):
#                 def thefunc(v):
#                     z = self._zsrf.ev(u, v)
#                     d = zht - z
#                     return d
#                 return thefunc
#             v = spo.brentq(intersectfunc(u), vi, vf, xtol=1e-6)
#             return v
# 
#         def _get_search_zones():
#             """Find the nearest v values above and below the plane
#             for each u value (for brentq limits).
#             """
#             # Check to see if we need to extend the set of
#             # surface points before looking for sign changes.
#             # (i.e. is zht above the tip or below the base?)
#             # Note that we *could* optimize this further by
#             # adjusting for each case separately, but I don't think
#             # it's worth the extra code.
#             if zht > rotxyzs[2,:,-1].min() or zht < rotxyzs[2,:,0].max():
#                 extvs = np.r_[self.bbox[2], self.bbox[3]]
#                 # Note (v, u) order in meshgrid(). This is to get
#                 # the expected array shape on evaluation later.
#                 V, U = np.meshgrid(extvs, self.u)
#                 U = U.ravel()
#                 V = V.ravel()
#                 extzs = self._zsrf.ev(U, V).reshape(len(self.u), len(extvs))
#                 vs = np.r_[extvs[0], self.v, extvs[1]]
#                 zs = np.c_[extzs[:,0], rotxyzs[2], extzs[:,1]]
#             else:
#                 vs = self.v
#                 zs = rotxyzs[2]
# 
#             # Subtract the zht from the surface z values. This
#             # will tell us which points are above/below the plane.
#             signs = np.sign(zs - zht)
#             # Fix the case where we hit the number exactly
#             signs[np.where(signs[:] == 0)] = 1
#             us, vcrossings = np.where(np.diff(signs) != 0)
#             vlimits = [[vs[vcndx], vs[vcndx+1]] for vcndx in vcrossings]
#             us = self.u[us]
#             return us, vlimits
# 
#         us, vlimits = _get_search_zones()
#         V = [_get_intersection_v(u, *vs) for u, vs in zip(us, vlimits)]
#         # Reset the z-surface coefficients to the original, unrotated
#         # values.
#         self._zsrf.tck = (self._zsrf.tck[0], self._zsrf.tck[1], coeffs[2])
#         pts = self.ev(us, V, mesh=False)
# 
#         return pts

    def plane_intersect_pts(self, planedef, extend_srf=False):
        """Given a plane definition (see below), returns the set of
        intersection points at each u-isospline.

        Parameters
        ----------
        planedef : array_like
            Normalized [A, B, C, D] values from the plane equation
            `Ax + By + Cz - D = 0`.

            As a reminder, A, B and C are the (normalized) values
            of the plane normal vector, and D is the dot product
            of *any* origin-to-plane vector and the normalized
            plane normal vector.

        extend_srf: bool, optional
            if `extend_srf` is True, intersections to the u-isosplines
            will be sought between self.v[0] and the lower end of
            the surface bounding box in the v-negative direction, as
            well as between self.v[-1] and the upper end of the surface
            bounding box in the v-positive direction. Note that this
            can have unexpected results when the surface bounding box
            is large and/or the surface extensions in v 'curl around'
            in an apparently unpredictable way (though of course, it
            is *totally* predictable!).
            If False, intersections are only sought between self.v[0]
            and self.v[-1], and if the plane is outside those bounds
            an error will be thrown.
        """
        import scipy.optimize as spo
        pnrml, D = planedef[0:3], planedef[3]
        def _get_intersection_v(u, vi=0.0, vf=1.0):
            def intersectfunc(u):
                def thefunc(v):
                    pt = self.ev(u, v) 
                    d = np.dot(pnrml, pt)
                    return d - D
                return thefunc
            v = spo.brentq(intersectfunc(u), vi, vf, xtol=1e-6)
            return v
        
    
        def _get_search_zones():
            """Find the nearest v values above and below the plane
            for each u value.
            """
            # Take the dot product between the vector to the plane
            # and each point on the surface, then subtract the D value
            # of the plane. This will tell us which points are above/below
            # the plane.
            if extend_srf:
                vs = np.array([self.bbox[2]] + list(self.v) + [self.bbox[3]])
                xyzs = self.ev(self.u, vs)
            else:
                vs = self.v
                xyzs = self.xyzs
 
            signs = np.sign(np.dot(xyzs.T, pnrml).T - D)
            # Fix the case where we hit the number exactly
            signs[np.where(signs[:] == 0)] = 2 
            us, vcrossings = np.where(np.diff(signs) != 0)
            vlimits = [[vs[vcndx], vs[vcndx+1]] for vcndx in vcrossings]
            us = self.u[us]
            return us, vlimits
     
        us, vlimits = _get_search_zones()
        V = [_get_intersection_v(u, *vs) for u, vs in zip(us, vlimits)]
        pts = self.ev(us, V, mesh=False)
        return pts
        
        

    def mplot(self, ures=1, vres=1, **kwargs):
        """Plot the surface using Mayavi's `mesh()` function

        Parameters
        ----------
        ures, vres: int
            Specifies the oversampling of the original
            surface in u and v directions. For example:
            if `ures` = 2, and `self.u` = [0, 1, 2, 3],
            then the surface will be resampled at
            [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
            plotting.

        kwargs: dict
            See Mayavi docs for `mesh()`

        Returns
        -------
            None
        """
        from enthought.mayavi import mlab

        # Set some Mayavi defaults
        _def_color = (0, 0, 1) # Blue
        if not kwargs.has_key('color'):
            kwargs['color'] = _def_color

        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        u, v = self.u, self.v
        lu, lv = len(u), len(v)
        nus = np.array(list(enumerate(u))).T
        nvs = np.array(list(enumerate(v))).T
        newundxs = np.linspace(0, lu - 1, ures * lu - (ures - 1))
        newvndxs = np.linspace(0, lv - 1, vres * lv - (vres - 1))
        hru = np.interp(newundxs, *nus)
        hrv = np.interp(newvndxs, *nvs)
        # Sample the surface at the new u, v values and plot
        meshpts = self.ev(hru, hrv, mesh=True)
        mlab.mesh(*meshpts, **kwargs)


if __name__ == '__main__':
#     from enthought.mayavi import mlab
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Set up a test surface (wavy cylinder)
    a = np.linspace(0, 2 * np.pi, 360)
    x, y = np.cos(a), np.sin(a)
    z = np.zeros(len(x)) # Seed value
    xyz = np.array([x, y, z])
    print xyz
    print xyz.shape
    xyz = np.array([xyz + i * np.array([[0, 0, .03]]).T
                    for i in range(200) ]).T.swapaxes(0, 1)
    f = 1.3 + .13 * np.sin(4 * np.linspace(0, 2 * np.pi, 200))
    print xyz
#     xyz[0:2,:,:] *= f
    print xyz.shape
    srf = ParaSurf(np.linspace(0, 1, len(x)),
                   np.linspace(0, 1, xyz.shape[2]), xyz,
                   bbox=[0.0, 1.0, -0.15, 1.15])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(*srf.xyzs,color="g")
    plt.show()
#     mlab.mesh(*srf.xyzs, color=(0, 1, 0), opacity=1.0)

    # Create a funky spiral around the surface and plot.
    u = np.linspace(0, 4, 4 * len(x)) % 1
    v = np.linspace(0, 1, 4 * len(y))
    pts = srf.ev(u, v, mesh=False)
    print pts[2]
    ax.plot(pts[0],pts[1],zs = pts[2].flatten(), color="r")
#     plt.show()
    #mlab.plot3d(*pts, tube_radius=0.02, color=(1, 1, 1)) # White line

    # Create a test plane and cut the surface with it
    ppt = np.array([0, 0, 2]) # Point on plane
    pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
    pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

    D = np.dot(ppt, pn)
    A, B, C = pn
    planedef = np.array([A, B, C, D])

    # Plot the plane
    def get_z(x, y): return (D - A * x - B * y) / C
    X, Y = 2 * np.mgrid[-1:1:2j, -1:1:2j]
    Z = get_z(X, Y)
    ax.plot_surface(X,Y,Z,color="b")
#     plt.show()
    #mlab.mesh(X, Y, Z, color=(0, 0, 1), opacity=0.5) # Blue plane

    # Get the plane-surface intersection and plot
    pipts = srf.plane_intersect_pts(planedef)
#     mlab.points3d(*pipts, scale_factor=0.03, color=(1, 0, 0))
    ax.plot(pipts[0],pipts[1],zs = pipts[2].flatten(), color="yellow")
    plt.show()
    # Plot a u-isospline on the surface, using the full
    # surface extensions
    V = np.linspace(srf.bbox[2], srf.bbox[3], 200)
    pts = srf.ev(0.0, V)
    #mlab.plot3d(*pts, tube_radius=0.02, color=(1, 1, 0)) # Yellow line
    #mlab.points3d(*pts, scale_factor=0.03, color=(1, 1, 0)) # Yellow dots
    ax.plot(pts[0],pts[1],zs = pts[2].flatten(), color="yellow")
    plt.show()
