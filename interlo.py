import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib.cm as cm

import astropy
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

import galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

class Objectset:
    def __init__(self, orbits):
        self.orbits = orbits
        self.num_objs = orbits.shape[0]

        self.time = None
        self.timesteps = None
        self.times = None

        self.integrated = False

    def integrate(self, time=5*u.Gyr, timesteps = 500):
        self.time = time
        self.timesteps=timesteps
        self.times = np.linspace(0, self.time.value, self.timesteps) * self.time.unit

        if self.integrated is False:
            self.orbits.integrate(self.times, MWPotential2014)

        self.integrated = True
        return self

    def plot_position(self, time = 0*u.Gyr, display = True):
        fig,axs = self.__setup_position_plot__()
        
        axs[0].scatter(self.orbits.x(time), self.orbits.y(time), s = 25)
        axs[1].scatter(self.orbits.r(time), self.orbits.z(time), s = 25)
        
        if display:
            plt.show()
        
        return fig,axs
    
    def animate_position(self, frames = None, figsize = (20,10), interval = 40, display = True):
        if frames is None:
            frames = self.timesteps

        fig, axs = self.plot_position(display=False)
        
        scat1 = axs[0].findobj(match=lambda x: isinstance(x, matplotlib.collections.PathCollection))[0]
        scat2 = axs[1].findobj(match=lambda x: isinstance(x, matplotlib.collections.PathCollection))[0]
        
        def update(time):
            xy = np.array([self.orbits.x(time),self.orbits.y(time)]).T
            scat1.set_offsets(xy)
            
            rz = np.array([self.orbits.r(time),self.orbits.z(time)]).T
            scat2.set_offsets(rz)
        
        skip = int(self.timesteps/frames)
        anim = animation.FuncAnimation(fig, update, frames = self.times[::skip], interval = interval)
        plt.close()

        if not display:
            return anim
        elif display:
            from IPython.display import HTML
            return HTML(anim.to_html5_video())
    
    def plot_orbit(self, visualize = True, figax = None, color = 'teal', alpha = .6, linewidth = .1):
        if figax is None:
            fig, axs = self.__setup_position_plot__()
        else:
            fig, axs = figax
        
        time = self.times
        
        axs[0].plot(self.orbits.x(time).T, self.orbits.y(time).T, alpha = alpha, linewidth = linewidth*2, color = color)
        axs[1].plot(self.orbits.r(time).T, self.orbits.z(time).T, alpha = alpha, linewidth = linewidth, color = color)
        
        if visualize:
            plt.show()
        
        return fig, axs

    def plot_radius(self, visualize = True, figax = None, color = 'teal', alpha = .6, linewidth = .1):
        if figax is None:
            fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (10,5))
        else:
            fig, axs = figax
        
        time = self.times
        print(self.orbits.r(time).T.shape)
        times = np.tile(self.times, (self.num_objs,1)).T
        axs.plot(times, self.orbits.r(time).T, alpha = alpha, linewidth = linewidth, color = color)
        
        if visualize:
            plt.show()
        else:
            return fig,axs

    def __setup_position_plot__(self, figsize = (20,10)):
        plt.rcParams["font.family"] = "sans"
        fig,axs = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        
        axs[0].set_xlabel("x Position", fontsize=16)
        axs[0].set_ylabel("y Position", fontsize=16)
        axs[0].set_xlim([-12,12])
        axs[0].set_ylim([-12,12])
        axs[0].set_aspect(1)
        
        axs[1].set_xlabel("r Position", fontsize=16)
        axs[1].set_ylabel("z Position", fontsize=16)
        #axs[1].set_xlim([6,12])
        axs[1].set_ylim([-1,1])
        #axs[1].set_aspect(3)
        
        return fig, axs

    def plotly_3d(self, time = 0*u.Gyr):
        import plotly
        import plotly.graph_objs as go
        #3d plotly plot of all objects
        x_arr = self.orbits.x(time)
        y_arr = self.orbits.y(time)
        z_arr = self.orbits.z(time)

        # Configure Plotly to be rendered inline in the notebook.
        plotly.offline.init_notebook_mode()

        # Configure the trace.
        trace = go.Scatter3d(
            x=x_arr,  # <-- Put your data instead
            y=y_arr,  # <-- Put your data instead
            z=z_arr,  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 4,
                'opacity': 0.6,
            }
        )

        # Configure the layout.
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )

        data = [trace]

        plot_figure = go.Figure(data=data, layout=layout)

        # Render the plot.
        plotly.offline.iplot(plot_figure)

class Starset(Objectset):
    def __init__(self, num_stars = 10, radial_dispersion = 12, azimuthal_dispersion = 11,\
                 z_dispersion = 9, asymmetric_drift = 5, r_extent = .5, z_extent = .5):
        self.num_objs = num_stars
        self.num_stars = num_stars
        self.radial_dispersion = radial_dispersion
        self.azimuthal_dispersion = azimuthal_dispersion
        self.z_dispersion = z_dispersion
        self.asymmetric_drift = asymmetric_drift
        self.r_extent = r_extent
        self.z_extent = z_extent
        self.orbits = self.__get_star_orbits__()
        
        self.integrated = False
        self.has_isos = False

        self.time = None
        self.times = None
        self.timesteps = None
        self.isos = None
        self.sun = None
        
        #self.stars = [Star(coord) for coord in zip(coords_6d)]
    
    def integrate(self, time=5*u.Gyr, timesteps = 500):
        self.time = time
        self.timesteps=timesteps
        self.times = np.linspace(0, self.time.value, self.timesteps) * self.time.unit

        if self.integrated is False:
            self.orbits.integrate(self.times, MWPotential2014)

        if self.has_isos is True:
            for isoset in self.isos:
                isoset.integrate(self.time, self.timesteps)

        self.integrated = True
    
    def get_isos(self, num_per_star = 100, v_eject = 1):
        #star.get_isos(num_per_star, eject_vel) for star in self.stars
        self.isos = np.empty(self.num_stars, dtype = ISOset)
        self.iso_orbits = np.empty(self.num_stars, dtype = Orbit)
        self.num_per_star = num_per_star
        for star in range(self.num_stars):
            self.isos[star] = ISOset(star = self.orbits[star], v_eject=v_eject)
        self.has_isos = True
    
    def get_sun(self):
        self.sun = Orbit()
        self.sun.integrate(self.times, MWPotential2014)

    def get_near_sun(self, time = 0*u.Gyr, distance = 1):
        x_sun = self.sun.x(time)
        y_sun = self.sun.y(time)
        z_sun = self.sun.z(time)
        near_sun_isos = np.empty((self.num_stars,self.num_per_star), dtype = bool)
        for i,isoset in enumerate(self.isos):
            x = isoset.orbits.x(time)
            y = isoset.orbits.y(time)
            z = isoset.orbits.z(time)
            near_sun_isos[i] = np.less(np.sqrt((x_sun-x)**2 + (y_sun-y)**2 + (z_sun-z)**2),distance)

        return near_sun_isos

    def get_rad_near_sun(self, time = 0*u.Gyr, distance = 1):
        near_sun = self.get_near_sun(time, distance)
        r = 0
        count = np.count_nonzero(near_sun)
        if count == 0:
            return None, 0
        for i,isoset in enumerate(self.isos):
            r += np.sum(isoset.orbits[near_sun[i]].r(0*u.Gyr))
        return r/count, count

    def plot_rad_near_sun(self, distance = 1, display = True, figax = None, color = 'teal', alpha = 1, linewidth = 1):
        if figax is None:
            fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (10,10))
        else:
            fig, axs = figax
        
        rad = np.empty(self.timesteps)
        counts = np.empty(self.timesteps)
        for i,time in enumerate(self.times):
            rad[i], counts[i] = self.get_rad_near_sun(time = time, distance = distance)
        axs[0].plot(self.times, rad, alpha = alpha, linewidth = linewidth, color = color)
        axs[1].plot(self.times, counts, alpha = alpha, linewidth = linewidth, color = color)
        
        axs[0].set_ylabel("Initial Radius Near Sun", fontsize=16)
        axs[1].set_ylabel("Number Near Sun", fontsize=10)
        axs[1].set_xlabel("Time", fontsize=10)

        if display:
            plt.show()
        else:
            return fig,axs

    def plot_iso_orbits(self, alpha = .02, linewidth = .1):
        figax = self.__setup_position_plot__()
        colors = cm.BuPu(np.linspace(.3, 1, self.num_stars))
        for i,isos in enumerate(self.isos):
            figax = isos.plot_orbit(visualize = False, figax = figax, color = colors[i], alpha = alpha, linewidth = linewidth)
        plt.show()

    #currently relies on celluloid. fix later.
    def animate_position(self, frames=None, figsize=(20, 10), interval=40, display=True):
        # anim = super().animate_position(frames, figsize, interval, display=False)

        # for isoset in self.isos
        #     iso_anim = isoset.animate_position(frames, figsize, interval, display=False)
        #     anim = self.zip_anim(anim, iso_anim)

        from celluloid import Camera

        fig, axs = self.__setup_position_plot__(figsize)
        colors = cm.BuPu(np.linspace(.3, 1, self.num_stars))
        camera = Camera(fig)
        for time in self.times:
            for i,isoset in enumerate(self.isos):
                axs[0].scatter(isoset.orbits.x(time), isoset.orbits.y(time), color = colors[i], s=10, alpha=.8)
                axs[1].scatter(isoset.orbits.r(time), isoset.orbits.z(time), color = colors[i], s=10, alpha=.8)
            if self.sun is not None:
                axs[0].scatter(self.sun.x(time), self.sun.y(time), s = 600, linewidths=2, facecolors='none', edgecolors='magenta',alpha = .7)
                axs[0].scatter(self.sun.x(time), self.sun.y(time), s = 8, color="magenta", alpha=.8)
                axs[1].scatter(self.sun.r(time), self.sun.z(time), s = 600, linewidths=2, facecolors='none', edgecolors='magenta',alpha = .7)
                axs[1].scatter(self.sun.r(time), self.sun.z(time), s = 8, color="magenta", alpha=.8)
            camera.snap()
        anim = camera.animate(interval = 40)

        if not display:
            return anim
        elif display:
            from IPython.display import HTML
            return HTML(anim.to_html5_video())

    def __get_star_orbits__(self):
        # Generate spherical polar coordinates (r, theta, phi)
        # Input positions
        n_points = self.num_stars
        radial_dispersion = self.radial_dispersion
        azimuthal_dispersion = self.azimuthal_dispersion
        z_dispersion = self.z_dispersion
        asymmetric_drift = self.asymmetric_drift
        r_extent = self.r_extent
        z_extent = self.z_extent
        
        r = np.random.uniform(8.5-r_extent, 8.5+r_extent, n_points) # 0 to 8kpc
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        z = np.random.uniform(-1*z_extent, z_extent, n_points) # -8kpc to 8kpc

        # Generate velocity dispersions
        radial_velocity = np.random.normal(0, radial_dispersion, n_points)
        
        galpy.potential.turn_physical_on(MWPotential2014) 
        circular_vel = galpy.potential.vcirc(MWPotential2014,r*u.kpc)
        galpy.potential.turn_physical_off(MWPotential2014) 
        
        azimuthal_velocity = np.random.normal(circular_vel, azimuthal_dispersion, n_points)
        z_velocity = np.random.normal(0, z_dispersion, n_points)
        asymmetric_drift_velocity = np.random.normal(0, asymmetric_drift, n_points)

        # Calculate spherical polar coordinates (r, theta, phi)
        phi = np.arccos(z / r)

        # Calculate Cartesian velocities
        x_velocity = radial_velocity * np.sin(phi) * np.cos(theta)\
                        + asymmetric_drift_velocity * np.sin(theta)\
                        - azimuthal_velocity * np.sin(theta)
        y_velocity = radial_velocity * np.sin(phi) * np.sin(theta)\
                        - asymmetric_drift_velocity * np.cos(theta)\
                        + azimuthal_velocity * np.cos(theta)
        z_velocity = radial_velocity * np.cos(phi) + z_velocity

        # Calculate Cartesian positions
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)

        coords = SkyCoord(x = x * u.kpc,\
                          y = y * u.kpc,\
                          z = z * u.kpc,\
                          v_x = x_velocity * (u.kilometer/u.second),\
                          v_y = y_velocity * (u.kilometer/u.second),\
                          v_z = z_velocity * (u.kilometer/u.second),\
                          representation_type='cartesian',\
                          differential_type=coord.CartesianDifferential,\
                          frame = 'galactocentric')
        
        return Orbit(coords) 

class ISOset(Objectset):
    def __init__(self, star, num_isos = 100, v_eject = 1):
        self._star = star
        self.num_isos = num_isos
        self.num_objs = num_isos
        self.v_eject = v_eject
        self.orbits = self.__get_orbits__()
        self.integrated = False

    def integrate(self, time=5*u.Gyr, timesteps = 500):
        self.time = time
        self.timesteps=timesteps
        self.times = np.linspace(0, self.time.value, self.timesteps) * self.time.unit
        if self.integrated is False:
            self.orbits.integrate(self.times, MWPotential2014)
        self.integrated = True

    def plot_ejection_velocities(self):
        #Plot directions ejected to show points are evenly distributed
        import plotly
        import plotly.graph_objs as go

        # Configure Plotly to be rendered inline in the notebook.
        plotly.offline.init_notebook_mode()

        # Configure the trace.
        trace = go.Scatter3d(
            x=self.orbits.vx() - self._star.vx(),  # <-- Put your data instead
            y=self.orbits.vy() - self._star.vy(),  # <-- Put your data instead
            z=self.orbits.vz() - self._star.vz(),  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 4,
                'opacity': 0.6,
            }
        )

        # Configure the layout.
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )

        data = [trace]

        plot_figure = go.Figure(data=data, layout=layout)

        # Render the plot.
        plotly.offline.iplot(plot_figure)

    def __get_orbits__(self):
        v = [self._star.vx(), self._star.vy(), self._star.vz()]
        r = np.array([self._star.x(), self._star.y(), self._star.z()])
        
        #get directions
        x = np.random.normal(size=self.num_isos)
        y = np.random.normal(size=self.num_isos)
        z = np.random.normal(size=self.num_isos)

        #normalize direction vectors
        norm = np.reciprocal(np.sqrt(np.square(x) + np.square(y) + np.square(z))) * self.v_eject
        v_eject = np.multiply(norm[:,np.newaxis],np.array([x,y,z]).T).T

        #add to star vel
        ones = np.ones(self.num_isos)
        v[0] = v[0] + v_eject[0]
        v[1] = v[1] + v_eject[1]
        v[2] = v[2] + v_eject[2]
        coords = SkyCoord(x = r[0]*ones * u.kpc,\
                          y = r[1]*ones * u.kpc,\
                          z = r[2]*ones * u.kpc,\
                          v_x = v[0] * (u.kilometer/u.second),\
                          v_y = v[1] * (u.kilometer/u.second),\
                          v_z = v[2] * (u.kilometer/u.second),\
                          representation_type='cartesian',\
                          differential_type=coord.CartesianDifferential,\
                          frame = 'galactocentric')
        
        return Orbit(coords)