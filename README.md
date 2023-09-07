# GPAD
__GPAD for [CARLA](https://github.com/carla-simulator/carla) PythonAPI__

## Forward
I finished my Ph.D. and I don't expect to spend a lot of time on this project anymore.
But I took some time to share it on GitHub just in case.

The code is not super self-explicit, 
but if you have questions or issues,
don't hesitate to open a new issue on the [GitHub page](https://github.com/renaudponcelet/GPAD), I would answer you
if I can :blush:.

If you want to use this code or the methods detailed in it, please cite me.
I first mentioned this code at __ICARCV2020__ 
in the following paper 
__Safe Geometric Speed Planning Approach for Autonomous Driving through Occluded Intersections.
R. Poncelet, A. Verroust-Blondet and F. Nashashibi. [[PDF]](https://hal.science/hal-02967740v2/document)__
```
@inproceedings{poncelet:hal-02967740,
  TITLE = {{Safe Geometric Speed Planning Approach for Autonomous Driving through Occluded Intersections}},
  AUTHOR = {Poncelet, Renaud and Verroust-Blondet, Anne and Nashashibi, Fawzi},
  URL = {https://hal.science/hal-02967740},
  BOOKTITLE = {{ICARCV 2020 - 16th International Conference on Control, Automation, Robotics and Vision}},
  ADDRESS = {Shenzhen, China},
  SERIES = {16th  International Conference on Control, Automation, Robotics and Vision},
  YEAR = {2020},
  MONTH = Dec
}
```

## Requirements

For now, this package is made to be cloned in the "PythonAPI"
folder in [CARLA](https://github.com/carla-simulator/carla)

It should run with Carla 0.9.14

You can check requirements.txt

## Structure

### Example
- main.py is an example to run the GPAD Planner detailed (in French) in my [Ph.D. thesis](https://renaudponcelet.zetmus.fr/these.pdf).

### Packages
- The Common package contains the GPAD Planner and the World manager and Utils.
- The Approaches package contains:
  - A subfolder [MMRIS](#MMRIS)
  - A subfolder [SGSPA](#SGSPA)
  - A subfolder Common

## Methods

### MMRIS
The MMRIS method is detailed (in French) in my [Ph.D. thesis](https://renaudponcelet.zetmus.fr/these.pdf).
This method is based on this paper (in English) [RIS](https://inria.hal.science/hal-01903318).
- RIS are computed for different speed profiles. I choose to compute three speed profiles : 
  - A brake speed profile
  - A constant speed profile
  - And an acceleration speed profile
- Paths are computed for each speed profile
- The "fastest" speed profile is selected

### SGSPA
The SGSPA is detailed in the following paper:
[Safe Geometric Speed Planning Approach for Autonomous Driving through Occluded Intersections](https://hal.science/hal-02967740v2)