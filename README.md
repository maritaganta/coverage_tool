# coverage_tool

## Purpose
Evaluate the performance of a satellite constellation in terms of target covered and revisit time.

## Necessary files: 

### main.py
Contains the functions and script for the coverage_tool()

### constellation_targets.csv
Csv file that contains the target list. It has three columns: 
1) Name of target
2) Latitude of target (in degrees)
3) Longitude of target (in degrees)
Number of rows == Number of targets

### constellation_matrix.npy
Numpy array that contains the satellite keplerian elements, with shape (number_of_satellites, 6). It has 6 columns:
1. Semimajor axis [km]
2. Eccentricity [0]
3. Inclination [deg]
4. Right ascension of the ascending node [deg]
5. Argument of Perigee [deg]
6. True anomaly [deg]

In the script each column will be loaded into a separate variable. 
Assumption (for script purposes): all satellites are in the same altitude -> the semimajor axis column must contain only one value

## Inputs in coverage_tool()
### keplerian elements
The constellation keplerian elements are imported through the constellation_matrix.npy file

### sensor characteristics
Assumption: all satellites have the same sensor.
The sensor is modeled thourgh two variables:
1. Across track angle f_acr [rad]
2. Along track angle f_alo [rad]

### date of simulation
In order to have the correct Earth's orientation with respect to the inertial frame the date of the simulation must be given (year, month, day). 
These will be used then to calculate the Julian day and consequently the Greenwich sidereal time.

### simulation length
The duration of the simulation (in seconds)

## Algorithm
1. creation of the access_profile matrix. It is a 2-d matrix, with shape (#targets, #timesteps). Each cell can have a value. If cell (i,j) == 0, then target i is not covered at timestep j.
   In case there is a passage, then the cell contains the id-number of the satellite that is hovering the target i at timestep j.
   a. From keplerian elements, the satellites are propagated in the timeframe (0-test_duration) using kep2car() function. It gives the satellites coordinates (x, y, z) in ECEF frame for each timestep
   b. The target list is transformed into a coordinates (x, y, z) list using latlon2car()
   c. The access_profile is build combining the satellite position vector, target position vector and the sensor characteristics inside the access_profile_function()
2. Using the covered_target_function() the covered targets are determined. It needs as input the access_profile matrix and the number of targets. 
   Then a sum of rows is performed. If a sum(row-i) == 0 then it means that target-i is not covered by the constellation
3. revisit_time_analytics() calculates the maximum, minimum and mean revisit time. 
   a. From the access_profile function a tuple of 2 arrays is calculated [covered_targets_ind, covered_timeslots_ind]: they contain the indices of the non-zero elements in access_profile
   b. For each target (row) then the differences between the indices of timesteps is calculated. They represents the gap in time between one satellite passage and another
   c. If a difference == 1, then it means there is a continuous coverage between the two successive timesteps
   d. max/min/mean(gap) represents the max/min/mean revisit time
   e. The three statistics are store in the revisit_time_analytics matrix for each target. At the end revisit_time_analytics has shape (#targets, 3)
