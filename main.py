#!/usr/bin/env python
# coding=utf-8

"""
Welcome to CARLA.
"""

from __future__ import print_function

import glob
import os
import sys

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import argparse
import logging
import json

from GPAD.Common.Utils.HUD import HUD
from GPAD.Common.Planner import Planner
from GPAD.Common.World import World

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    carla_world = None
    client = None
    fps = 20.0
    i = args.i
    count = args.i
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        scenario_path = 'Scenarios/'
        if args.scenario_mode:
            scenario = args.scenario_name
            super_scenario_data = {}
            if '/' in scenario:
                super_scenario_file_path = scenario_path + scenario[:scenario.find('/')] + '/data.json'
                try:
                    with open(super_scenario_file_path) as f:
                        super_scenario_data = json.load(f)
                except FileNotFoundError:
                    super_scenario_data = {}
            scenario_file_path = scenario_path + scenario + '/data.json'
            with open(scenario_file_path) as f:
                scenario_data = json.load(f)
            if "map" in scenario_data:
                world_map = scenario_data["map"]
            elif "map" in super_scenario_data:
                world_map = super_scenario_data["map"]
            else:
                raise Exception("the \"map\" argument cannot be found")
        else:
            if args.map is None:
                raise Exception("please indicate a map name (-m map_name) or a scenario (-s scenario_file)")
            world_map = args.map
        carla_world = client.load_world(world_map)
        if world_map[-3:] == 'Opt':
            carla_world.unload_map_layer(carla.MapLayer.All)
        if client.load_world(world_map).get_map().name != "Carla/Maps/" + world_map:
            print(client.get_available_maps())
            raise Exception("the map is not the wanted one")

        clock = pygame.time.Clock()
        hud = HUD(args.width, args.height)
        world = World(carla_world, client, hud, clock, args, fps)
        print("world is set")
        planner = Planner(world, client, args.autopilot, fps)
        world.set_planner(planner)
        print("planner is set")
        if args.rec:
            scenario_name = args.scenario_name
            if scenario_name is None:
                scenario_name = 'temp'
            if '/' in scenario_name:
                scenario_name = scenario_name.replace('/', '_')
            client.start_recorder(scenario_name + ".log")
            print("recording is start")
        while True:
            carla_world.tick()
            if planner.parse_events(clock):
                if count > 0:
                    count -= 1
                else:
                    return
                delay = args.delay
                if args.wd:
                    delay = args.delay * (1 - (i - count) / i)
                world.respawn(delay)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    except Exception as e:
        print(e)
    finally:
        if args.rec:
            client.stop_recorder()
        settings = carla_world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        carla_world.apply_settings(settings)
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main(args):
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except Exception as e:
        if e == KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
        else:
            print(e)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Personal arguments
    parser.add_argument('-a', '--auto', action="store_true", default=False,
                        dest="auto_mode", help="Ego is in autopilot by default, use -d, --delay [time in ms] to set a"
                                               "delay")
    parser.add_argument('-t', '--time_delay', action="store", default=0.0, dest="delay",
                        help="Set a delay until autopilot is set")
    parser.add_argument('-i', '--iterations', action="store", default=0, dest="i", help="Number of iterations")
    parser.add_argument('-w', '--wane_delay', action="store_true",
                        default=False, dest="wd", help="The delay is reduced as iterations progress")
    parser.add_argument('-l', '--planner_list', action='append',
                        default=[], dest="planner_list", help="Precise which planner you want to use (ris-path, "
                                                              "vis-speed)")
    parser.add_argument('-d', '--display', action="store_true", default=False,
                        dest="display", help="Display environment and planner results in the _out directory")
    parser.add_argument('-r', '--record', action="store_true", default=False,
                        dest="rec", help="Record ego_vehicle information in the _out directory")
    parser.add_argument('-s', '--scenario_name', action="store", default=None, dest="scenario_name",
                        help="Set the scenario name")
    parser.add_argument('-rs', '--rec_scenario', action="store_true", default=False,
                        dest="rec_scenario", help="Record the scenario in Scenarios/temp/")
    parser.add_argument('-aw', '--awareness', action="store", default='omniscient',
                        dest="awareness", help="approach option")
    parser.add_argument('-cs', '--cross_sec', action="store_true", default=False,
                        dest="cross_sec", help="approach option")
    parser.add_argument('-m', '--map', action="store", default=None,
                        dest="map", help="map")
    parser.add_argument('-at', '--allow_threading', action="store_true", default=False,
                        dest="allow_threading", help="computation option")
    # Carla arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    parser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    parser.add_argument(
        '--autopilot',
        action='store_true',
        help='enable autopilot')
    parser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 640x480)')

    results = parser.parse_args()
    if results.scenario_name is None:
        results.scenario_mode = False
    else:
        results.scenario_mode = True
    if len(results.planner_list) == 0:
        raise Exception("You have to choose at least one planner, the implemented ones are in Approaches directory")
    results.delay = float(results.delay)
    results.i = int(results.i)
    main(results)
