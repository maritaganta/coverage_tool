import numpy as np
from numpy import linalg as LA
import time
import cProfile


# Constants
deg = np.pi / 180  # converts degrees to radiant
mu = 398600.44  # [km3/s2] gravitational parameter
J2 = 0.00108263  # [-] second zonal harmonic
Re = 6378.14  # [km] earth's radius
we = (2 * np.pi + 2 * np.pi / 365.26) / (24 * 3600)  # [rad/s] earth's angular velocity
exp = np.exp(1)


# gives position and velocity vector at each time step (thus determining where the satellite is at any point)
def keplerian_to_cartesian(sma, ecc, inc: np.array, raan: np.array, aop, ta, test_duration, t_length=1):
    inc = np.array([inc])
    raan = np.array([raan])
    # v_ones_i = np.ones(inc.shape)
    h = np.sqrt(mu * sma * (1 - (ecc ** 2)))
    e_0 = 2 * np.arctan(np.tan(ta / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # [rad] initial eccentric anomaly
    m_0 = e_0 - ecc * np.sin(e_0)
    v_ones_raan = np.ones(raan.shape)

    fac = -3 / 2 * np.sqrt(mu) * J2 * (Re ** 2) / ((1 - (ecc ** 2)) ** 2) / (sma ** 7 / 2)
    raandot = fac * np.cos(inc)
    aopdot = fac * (5 / 2 * (np.sin(inc) ** 2) - 2)
    Mdot = -3 / 2 * J2 * (Re / (sma * (1 - (ecc ** 2))) ** 2) * np.sqrt(1 - (ecc ** 2)) * (
            3 / 2 * (np.sin(inc) ** 2) - 1)
    Mdott = np.sqrt(mu / (sma ** 3)) * (1 + Mdot)
    W_dot = raandot * v_ones_raan

    # Period
    # the orbit period T
    T = 2 * np.pi / (aopdot + Mdott)  # [s] period of the orbit
    # if I have more than one period
    # n_periods = test_duration / T
    T_r = test_duration

    # time to/from perigee
    to = m_0 * (T / (2 * np.pi))  # [s] initial time for the ground track
    # final time
    tf = to + T_r  # [s] final time for the ground track

    # take the time step count
    n_step = np.ceil(T_r / t_length).astype(int)  # [steps]
    n_step_max = np.max(n_step)
    step_l = T_r / n_step_max

    # Timeline
    # create timeline
    times = np.array([np.linspace(to, tf, n_step_max)])

    # propagation through time
    M = m_0 + Mdott * times  # theta
    W = raan + W_dot * times  # RAAN

    wp = aop + aopdot * times  # AoP

    # vector of shape times
    E = kepler_E(ecc, M)  # ECC anomaly

    # true anomaly at each timestep
    TA = 2 * np.arctan(np.tan(E / 2) * np.sqrt((1 + ecc) / (1 - ecc)))

    # position of the satellite in peri-focal frame
    r = (h ** 2) / mu / (1 + ecc * np.cos(TA)) * np.array([np.cos(TA), np.sin(TA), 0 * np.ones(TA.shape)])
    v = mu / h * np.array([-np.sin(TA), ecc + np.cos(TA), 0 * np.ones(TA.shape)])

    # DCM from peri-focal to ECI
    Q_xX = np.array([[-np.sin(W) * np.cos(inc) * np.sin(wp) + np.cos(W) * np.cos(wp),
                      -np.sin(W) * np.cos(inc) * np.cos(wp) - np.cos(W) * np.sin(wp), np.sin(W) * np.sin(inc)],
                     [np.cos(W) * np.cos(inc) * np.sin(wp) + np.sin(W) * np.cos(wp),
                      np.cos(W) * np.cos(inc) * np.cos(wp) - np.sin(W) * np.sin(wp), -np.cos(W) * np.sin(inc)],
                     [np.sin(inc) * np.sin(wp) * v_ones_raan, np.sin(inc) * np.cos(wp) * v_ones_raan,
                      np.cos(inc) * np.ones(wp.shape) * v_ones_raan]])

    # R,V are the position and velocity vector in ECI
    if np.ndim(r) == 5:
        R = np.einsum('mnusoi,npsri->mois', Q_xX, r)
        V = np.einsum('mnusoi,npsri->mois', Q_xX, v)
    elif np.ndim(r) == 4:
        R = np.einsum('mnuso,npsr->ms', Q_xX, r)
        V = np.einsum('mnuso,npsr->ms', Q_xX, v)
    else:
        R = np.einsum('mnus,nps->ms', Q_xX, r)
        V = np.einsum('mnus,nps->ms', Q_xX, v)

    # how much the earth has rotated during this time
    theta = we * (times - to)

    # DCM ECI from ECEF
    Q = np.array([[np.cos(theta), np.sin(theta), 0 * np.ones(theta.shape)],
                  [-np.sin(theta), np.cos(theta), 0 * np.ones(theta.shape)],
                  [0 * np.ones(theta.shape), 0 * np.ones(theta.shape), 1 * np.ones(theta.shape)]])

    if np.ndim(r) == 5:
        r_rel = np.einsum('mnuswi,nois->mois', Q, R)
        v_rel = np.einsum('mnuswi,nois->mois', Q, V)
    elif np.ndim(r) == 4:
        r_rel = np.einsum('mnuso,ns->ms', Q, R)
        v_rel = np.einsum('mnuso,ns->ms', Q, V)
    else:
        r_rel = np.einsum('mnus,ns->ms', Q, R)
        v_rel = np.einsum('mnus,ns->ms', Q, V)

    return r_rel, v_rel, step_l, T, n_step_max


def kep2car(a, e, incl, W_o, wpo, TAo, test_duration, t_length, year, month, day):
    # Propagates the orbit over the specified time interval, transforming
    # the position and velocity vectors into the earth-fixed frame

    Eo = 2 * np.arctan(np.tan(TAo / 2) * np.sqrt((1 - e) / (1 + e)))  # [rad] initial eccentric anomaly
    Mo = Eo - e * np.sin(Eo)  # [rad] initial mean anomaly

    v_ones_W = np.ones(W_o.shape)   # support vector in order to match the dimensions inside the transformation matrix

    p = a * (1 - e ** 2)     # semi-latus rectum

    # J2-perturbation of keplerian elements
    wpdot = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    Wdot = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)
    Mdott = np.sqrt(mu / a ** 3) * (
            1 - 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(1 - e ** 2) * (3 / 2 * np.sin(incl) ** 2 - 1))

    om_d = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    OM_d = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)

    W_dot = Wdot * v_ones_W

    T = 2 * np.pi / (wpdot + Mdott)  # [s] period of the orbit
    T_r = test_duration   # [s]

    to = Mo * (T / (2 * np.pi))  # [s] initial time for the ground track
    tf = to + T_r  # [s] final time for the ground track

    n_step = np.ceil(T_r / t_length).astype(int)  # [steps]
    n_step_max = np.max(n_step)
    step_l = T_r / n_step_max       # step length

    times = np.array([np.linspace(to, tf, n_step_max)])  # [s] times at which ground track is plotted
    M = Mo + Mdott * times   # Mean anomaly
    W = W_o + W_dot * times  # RAAN

    wp = wpo + wpdot * times    # argument of perigee
    E = kepler_E(e, M)          # eccentric anomaly
    TA = 2 * np.arctan(np.tan(E / 2) * np.sqrt((1 + e) / (1 - e)))   # true anomaly

    nu = TA
    om = wpo + om_d * times

    g0 = g0_fun(year, month, day)       # calculates the greenwich sidereal time

    th = we * (times - to) + g0         # calculates the earth angle rotation at given time
    OM = W_o + OM_d * times             # raan

    const = (a * (1 - e ** 2)) / (1 + e * np.cos(nu))    # constant

    # Calculate position coordinates x, y, z with transformation matrix
    r1 = const * (np.cos(incl) * np.sin(nu + om) * np.sin(th - OM) + np.cos(nu + om) * np.cos(th - OM))
    r2 = const * (np.cos(incl) * np.sin(nu + om) * np.cos(th - OM) - np.cos(nu + om) * np.sin(th - OM))
    r3 = const * (np.sin(incl) * np.sin(nu + om) * v_ones_W)

    r = np.moveaxis(np.concatenate((r1, r2, r3), axis=0), 1, -1)  # moves order components

    # Calculate velocity coordinates x, y, z with transformation matrix
    v1 = const * ((Mdott + wpdot) * (
            np.cos(incl) * np.cos(nu + om) * np.sin(th - OM) - np.sin(nu + om) * np.cos(th - OM)) + (we - OM_d) * (
                          np.cos(incl) * np.sin(nu + om) * np.cos(th - OM) - np.cos(nu + om) * np.sin(th - OM)))
    v2 = const * ((Mdott + wpdot) * (
            np.cos(incl) * np.cos(nu + om) * np.cos(th - OM) + np.sin(nu + om) * np.sin(th - OM)) - (we - OM_d) * (
                          np.cos(incl) * np.sin(nu + om) * np.sin(th - OM) + np.cos(nu + om) * np.cos(th - OM)))
    v3 = const * ((Mdott + wpdot) * (np.sin(incl) * np.cos(nu + om) * v_ones_W))

    v = np.moveaxis(np.concatenate((v1, v2, v3), axis=0), 1, -1)

    return r, v, step_l, W, wp, TA, th, M, times


def j0_fun(year, month, day):
    # calculates the julian day number at 0 UT

    j0 = 367 * year - np.fix(7 * (year + np.fix((month + 9) / 12)) / 4) + np.fix(275 * month / 9) + day + 1721013.5

    return j0


def g0_fun(year, month, day):  # calculates the greenwich sidereal time
    j0 = j0_fun(year, month, day)

    j = (j0 - 2451545) / 36525

    g0 = 100.4606184 + 36000.7704 * j + 0.000387933 * (j ** 2) - 2.583 * (10 ** -8) * (j ** 3)

    g0 = np.radians(g0)

    nn = np.floor(g0 / (2 * np.pi))

    g0 = g0 - nn * 2 * np.pi

    return g0


def kepler_E(e, M):     # convertion from mean to eccentric anomaly
    error = 1e-8  # Error tolerance
    E = np.ones(M.shape)
    ratio = np.ones(M.shape)

    # Select starting value for E
    E[:] = M - e / 2
    E[M < np.pi] = M[M < np.pi] + e / 2
    while np.abs(np.max(ratio)) > error:
        ratio = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - ratio
    return E


# this gives two vectors (lat and lon) of the ground track in time steps
# and therefore allows for the ground track to be plotted.
# first and last steps will be the starting and ending point
def dec_and_ra_from_r(r):
    l = r[0, :] / LA.norm(r, axis=0)  # direction cosine
    m = r[1, :] / LA.norm(r, axis=0)  # direction cosine
    n = r[2, :] / LA.norm(r, axis=0)  # direction cosine
    dec = np.arcsin(n)
    ra = 2 * np.pi - np.arccos(l / np.cos(dec))
    ra[m > 0] = np.arccos(l[m > 0] / np.cos(dec[m > 0]))
    dec = np.degrees(dec)  # latitude (degrees)
    ra = np.degrees(ra)  # long (degrees)
    return dec, ra  # first and lasts indices give the starting and ending positions


# returns a vector of True and False for the access and
# a vector of the targets accessed and duration of visibility in time steps
def in_out(a, r, v, r_t, f_acr, f_alo):
    eta = a / Re
    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
    p1, p2, p3 = projections(r, v, r_t)

    # along track
    psi = np.arctan2(p2, p1)
    filt_steps_al = np.absolute(psi) <= a_beta

    # across track
    phi = np.arctan2(p3, p1)
    filt_steps_ac = np.absolute(phi) <= a_alfa

    filt_steps = np.logical_and(filt_steps_al,
                                filt_steps_ac)  # np.array - boolean - shape: (#targets, #timesteps). Timeline, targets POV
    filt_targets = np.any(filt_steps,
                          axis=1)  # np.array - boolean - shape: (1, #targets). Tells whether a target is covered or not
    cov_steps = np.array(np.nonzero(filt_steps[:]))
    # n_targets = r_t.shape[1]  # number of targets

    return filt_steps, filt_targets, cov_steps


def projections(r, v, r_t):
    u_r = unit_v(r)
    u_v = unit_v(v)
    u_r_t = unit_v(r_t)

    u_h = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = u_h / LA.norm(u_h, axis=0)
    u_y = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = u_y / LA.norm(u_y, axis=0)

    # target projection on new system of reference
    p1 = dot_p(u_r, u_r_t)
    p2 = dot_p(u_y, u_r_t)
    p3 = dot_p(u_h, u_r_t)

    # p1 = np.dot(u_r, u_r_t)
    # p2 = np.dot(u_y, u_r_t)
    # p3 = np.dot(u_h, u_r_t)

    return p1, p2, p3


# transforms from geodetic to cartesian taking into account the earth's radius
# to get the position vector of each target
# Accepts longitude in the range of (0, 360) so lon = lon + 180 on the outer scope
def latlon2car(lon, lat, R):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    r = np.array([x, y, z])
    return r


def read_targets():
    lon_t, lat_t = np.loadtxt("constellation_targets.csv", delimiter=',', usecols=(1, 2), unpack=True)
    # lon_t += 180
    return lon_t, lat_t


def unit_v(v):
    u_v = v / LA.norm(v, axis=0)  # direction cosine
    return u_v


def dot_p(r_sat, r_t):
    ang = np.einsum('mns,mt->ts', r_sat, r_t)
    return ang


def gt_r(a, r, v, f_acr, f_alo):
    eta = a / Re
    a_beta = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = - f_alo + np.arcsin(eta * np.sin(f_alo))

    u_r = unit_v(r)
    u_v = unit_v(v)

    u_h = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = u_h / LA.norm(u_h, axis=0)
    u_y = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = u_y / LA.norm(u_y, axis=0)

    r_uf = np.cos(a_beta) * np.cos(-a_alfa) * u_r - np.sin(-a_alfa) * u_y + np.sin(a_beta) * np.cos(-a_alfa) * u_h
    r_df = np.cos(-a_beta) * np.cos(-a_alfa) * u_r - np.sin(-a_alfa) * u_y + np.sin(-a_beta) * np.cos(-a_alfa) * u_h
    r_ub = np.cos(a_beta) * np.cos(a_alfa) * u_r - np.sin(a_alfa) * u_y + np.sin(a_beta) * np.cos(a_alfa) * u_h
    r_db = np.cos(-a_beta) * np.cos(a_alfa) * u_r - np.sin(a_alfa) * u_y + np.sin(-a_beta) * np.cos(a_alfa) * u_h

    return r_uf, r_ub, r_df, r_db


# will return lon/lat for the bounding box of each time step
def gt_lat_lon(r_uf, r_ub, r_df, r_db):
    lon_uf, lat_uf = dec_and_ra_from_r(r_uf)
    lon_ub, lat_ub = dec_and_ra_from_r(r_ub)
    lon_df, lat_df = dec_and_ra_from_r(r_df)
    lon_db, lat_db = dec_and_ra_from_r(r_db)

    return lon_uf, lat_uf, lon_ub, lat_ub, lon_df, lat_df, lon_db, lat_db


def target_visibility(a, r, v, r_t, f_acr, f_alo, Re, ratio):
    eta = a / Re  # ratio between altitude and Earth radius
    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))  # [rad] sensor angles from Earth center
    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))  # [rad] sensor angles from Earth center

    n_timesteps = len(r[1])  # number of total time-steps in the simulations
    n_scenes = np.floor(ratio * np.pi / a_beta).astype(int)  # number of scenes (assuming e =0, spherical earth)
    l_scenes = np.floor(n_timesteps / n_scenes).astype(int)  # length of every scene (in time-steps)

    accessibility = []  # list containing the accessible targets for each scene
    accessible_targets = []  # list containing all the targets visible in the simulation

    n_targets = r_t.shape[1]  # number of targets
    target_index = np.arange(n_targets)  # vector containing the indexes of all targets

    r_index_vec = np.empty(n_scenes)  # vector containing the index of time-steps related to each scene
    r_index = 0  # initial value for timestep index (first scene)

    for j in np.arange(n_scenes):
        r_scene = r[:, r_index]  # choosing the appropriate position vector for this scene
        v_scene = v[:, r_index]  # choosing the appropriate velocity vector for this scene

        # some calculations for determine whether the targets are in or out the scene
        u_r = unit_v(r_scene)
        u_v = unit_v(v_scene)
        u_r_t = unit_v(r_t)
        # print(u_r.shape, u_v.shape, u_r_t.shape)
        # print('r_t', r_t)
        u_h = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
        u_h = u_h / LA.norm(u_h, axis=0)
        u_y = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
        u_y = u_y / LA.norm(u_y, axis=0)

        p1 = np.dot(u_r, u_r_t)
        p2 = np.dot(u_y, u_r_t)
        p3 = np.dot(u_h, u_r_t)

        # along track
        psi = np.arctan2(p2, p1)

        filt_steps_al = np.absolute(psi) <= a_beta

        # across track
        phi = np.arctan2(p3, p1)

        filt_steps_ac = np.absolute(phi) <= a_alfa
        #######
        # boolean array (1,n_targets) containing whether the target is in the scene or not
        filt_steps = np.logical_and(filt_steps_al,
                                    filt_steps_ac)

        r_index_vec[j] = r_index  # saving the current timestep index to the vector of scenes indexes
        r_index = r_index + l_scenes  # updating the timestep index
        r_index = r_index[0]

        acc_targets = target_index[filt_steps].tolist()  # vector containing the accessible targets for the scene

        accessibility.append(acc_targets)  # saving the accessible targets for the current scene

        if np.any(filt_steps):
            accessible_targets.append(acc_targets)  # if there are visible targets, save them

    return accessibility, accessible_targets, r_index_vec, n_scenes, l_scenes


def bounding_b(r_index_vec, lon_uf, lat_uf, lon_ub, lat_ub, lon_df, lat_df, lon_db, lat_db):
    lo_uf = lon_uf[r_index_vec.astype(int)]  # extracting the appropriate coordinates (scenes)
    lo_ub = lon_ub[r_index_vec.astype(int)]
    lo_df = lon_df[r_index_vec.astype(int)]
    lo_db = lon_db[r_index_vec.astype(int)]

    la_uf = lat_uf[r_index_vec.astype(int)]
    la_ub = lat_ub[r_index_vec.astype(int)]
    la_df = lat_df[r_index_vec.astype(int)]
    la_db = lat_db[r_index_vec.astype(int)]

    longitude = []  # initializing vectors
    latitude = []

    n_scenes = r_index_vec.shape[0]

    for j in np.arange(n_scenes):
        coordinates_lo = [lo_uf[j], lo_df[j], lo_db[j], lo_ub[j]]
        coordinates_la = [la_uf[j], la_df[j], la_db[j], la_ub[j]]

        longitude.append(coordinates_lo)
        latitude.append(coordinates_la)

    return longitude, latitude


def access_profile_function(lat_t, lon_t, n_step, n_sat, sma, ecc, inc, raan, aop, ta, test_duration, t_length, year,
                            month, day, f_acr, f_alo):
    r_t = latlon2car(lat_t, lon_t, Re)  # [km] transform target lat/lon into x,y,z coordinates
    n_tar = r_t.shape[1]  # [-] number of targets

    access_profile = np.zeros((n_tar, n_step),
                              dtype=int)  # initialize target access profiles. array shape (#targets, #timesteps)

    for sat_n in np.arange(n_sat):
        r, v, step_l, W, wp, TA, th, M, times = kep2car(sma, ecc[sat_n], np.array([inc[sat_n]]),
                                                        np.array([raan[sat_n]]), aop[sat_n], ta[sat_n], test_duration,
                                                        t_length, year,
                                                        month,
                                                        day)  # propagate constellation satellites in the simulation time

        filt_steps, filt_targets, cov_steps = in_out(sma, r, v, r_t, f_acr,
                                                     f_alo)  # determine the coverage for each timestep

        access_profile[filt_steps] = sat_n + 1  # set satellite passage in the target access profiles

    return access_profile, n_tar


def covered_target_function(access_profile, n_tar):
    covered_target_filter = np.sum(access_profile, axis=1)  # sum of the rows. If cell == 0 then no coverage
    target_list = np.arange(
        n_tar) + 1  # target list id. First target is identified with 1, last target is identified with n_target + 1
    covered_target = target_list[covered_target_filter != 0]  # determine whether the target is covered or not
    print('covered targets id: ', covered_target)

    return covered_target


def revisit_time_analytics_function(access_profile, n_tar, t_length):
    covered_tuples = np.nonzero(access_profile)  # return tuple of 2 arrays, one for each dimension of access_profile. They contain the indices of covered timeslots.
    # First array: covered targets. Second array: covered timeslot

    covered_targets_ind = covered_tuples[0]    # array containing the indices of covered targets
    covered_timeslots_ind = covered_tuples[1]  # array containing the indices of covered timeslots

    covered_targets = np.unique(covered_targets_ind)  # array containing the list of covered targets

    revisit_time_analytics = np.zeros((n_tar,
                                       3))  # initialize array with shape (#n_targets, 3) where: 1 col: max_revTime, 2 col: min_revTime, 3 col: mean_revTime

    for j in covered_targets:
        timeline_filter = covered_targets_ind == j  # boolean filter. Help filter the timeline covered slots to the specific current target
        current_timeline = covered_timeslots_ind[timeline_filter]  # covered timeslots for the current target

        gaps = np.diff(
            current_timeline)  # gives the difference between each timeslot cell value. If cell != 1 then there is a gap in time

        max_gap = np.amax(gaps) * t_length  # [sec]
        min_gap = np.amin(gaps[gaps != 1]) * t_length  # [sec] disregard continuous access
        mean_gap = np.mean(gaps[gaps != 1]) * t_length  # [sec] disregard continuous access

        print('target', j + 1)     # print target id
        print('max_gap: ', np.around(max_gap / 60, 2), 'min')        # maximum revisit time
        print('min_gap: ', np.around(min_gap, 2), 'sec')             # minimum revisit time
        print('mean_gap: ', np.around(mean_gap / 60, 2), 'min \n')   # mean revisit time

        revisit_time_analytics[j, 0] = max_gap    # store in array
        revisit_time_analytics[j, 1] = min_gap
        revisit_time_analytics[j, 2] = max_gap

    return revisit_time_analytics


def snapshot_analytics_function(snapshot_time, t_length, access_profile, n_tar):
    snapshot_timestep = np.floor(snapshot_time / t_length).astype(int)  # [-] conversion of snapshot_time into the corresponding timestep

    restricted_access_profile = access_profile[:, snapshot_timestep:-1]  # restrict the access profile matrix by having as first timestep the snapshot_timestep
    restricted_access_profile_bool = restricted_access_profile != 0  # transform the restricted access profile into 0/1 where 0=no coverage, 1=coverage

    first_sat_pass_timestep = restricted_access_profile_bool.argmax(axis=1)  # obtain the indices where the maximum value along the row (target) is !=0 (coverage). Resulting array would be (#targets, 1). It represents also how many timesteps from snapshot_step is the next satellite passage. If one cell = 0, then there is currently a satellite passage
    first_sat_pass_time = first_sat_pass_timestep * t_length  # obtain the time_gap in seconds
    first_sat_pass_id = restricted_access_profile[np.arange(n_tar), first_sat_pass_timestep]  # obtain the array of the satellite's id of next first passage

    print('Next satellite passage in [sec]: ', first_sat_pass_time)
    print('Next satellite id: ', first_sat_pass_id)

    return first_sat_pass_time, first_sat_pass_id


def coverage_tool():
    # ---- INPUTS ----
    # keplerian elements
    constellation_matrix = np.load('constellation_matrix.npy')  # load constellation matrix. shape (#sat, 6)
    sma = constellation_matrix[0, 0]  # [km] semimajor axis
    ecc = constellation_matrix[:, 1]  # [-]  eccentricity
    inc = constellation_matrix[:, 2] * deg  # [rad] inclination
    raan = constellation_matrix[:, 3] * deg  # [rad] right ascention of the ascending node
    aop = constellation_matrix[:, 4] * deg  # [rad] argument of perigee
    ta = constellation_matrix[:, 5] * deg  # [rad] true anomaly

    n_sat = inc.shape[0]  # [-] number of satellite in the constellation

    # sensor characteristics
    f_acr = 31 * deg  # [rad] across track angle
    f_alo = 16 * deg  # [rad] along track angle

    # date of simulation
    year = 2022
    month = 4
    day = 1

    # simulation length
    test_duration = 85522  # [sec]
    t_length = 1  # [sec]  duration of each timestep in seconds
    n_step = np.ceil(test_duration / t_length).astype(int)  # [steps]  calculate how many timesteps in timeline

    # target import
    lon_t, lat_t = read_targets()  # [deg] longitude, latitude

    # snapshot analytics function
    snapshot_time = 15000  # [sec]  time after which I want to know when it is the next satellite passage and by which satellite

    # ---- End Inputs ----

    # Determine the target access profile matrix
    access_profile, n_tar = access_profile_function(lat_t, lon_t, n_step, n_sat, sma, ecc, inc, raan, aop, ta,
                                                    test_duration, t_length, year, month, day, f_acr, f_alo)

    # covered target function
    covered_target = covered_target_function(access_profile, n_tar)

    # revisit time analytics function
    revisit_time_analytics = revisit_time_analytics_function(access_profile, n_tar, t_length)

    # snapshot analytics function
    print('snapshot time: ', snapshot_time, '[sec]')
    first_sat_pass_time, first_sat_pass_id = snapshot_analytics_function(snapshot_time, t_length, access_profile, n_tar)

    return access_profile, covered_target, revisit_time_analytics, first_sat_pass_time, first_sat_pass_id


# Grab Currrent Time Before Running the Code
start = time.time()

access_profile, covered_target, revisit_time_analytics, first_sat_pass_time, first_sat_pass_id = coverage_tool()

# Grab Currrent Time After Running the Code
end = time.time()
# Subtract Start Time from The End Time
total_time = end - start
print("\n execution time: " + str(total_time) + " seconds")

cProfile.run('coverage_tool()')