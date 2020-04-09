import random
import itertools
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors


import constants
from utils import drawing
from utils import game_util
from generate_questions.episode import Episode
from graph import graph_obj

import copy
import math
from PIL import Image, ImageDraw
from ai2thor.controller import Controller

#np.set_printoptions(threshold=sys.maxsize)

DEBUG = True
SIM_TIMES = constants.RANDOM_SIMULATION_TIME

def get_agent_map_data(c: Controller):
    c.step({"action": "ToggleMapView"})
    frame = c.last_event.frame
    c.step({"action": "ToggleMapView"})
    return frame

def main():
    scene_numbers = constants.TRAIN_SCENE_NUMBERS
    scene_numbers = list(range(1, 31))
            
    def create_dump():
        episode = Episode()
        count = 0
        #for i in range(len(scene_numbers)):
        while count > -1: 
            scene_name = 'FloorPlan%d' % scene_numbers[count]
            print ('scene is %s' % scene_name)
            grid_file = 'layouts/%s-layout.npy' % scene_name
            graph = graph_obj.Graph(grid_file)#, use_gt=use_gt)
            print ("graph: ", graph)
            scene_bounds = [graph.xMin, graph.yMin,
                graph.xMax - graph.xMin + 1,
                graph.yMax - graph.yMin + 1]
            print ("bounds: ", scene_bounds)
            episode.initialize_scene(scene_name)
            #scene_seed = random.randint(0, 999999999)
            #episode.initialize_episode(scene_seed=scene_seed)  # randomly initialize the scene
            
            prefix = 'object_semantic_map/maps/'
            for index in range(1, 11):
            #memory = np.zeros((graph.yMax - graph.yMin + 1, graph.xMax - graph.xMin + 1, 1 + constants.NUM_CLASSES))
                npy_path = os.path.join(prefix, str(index)+'.npy')
                memory = np.load(npy_path)
                print (memory.shape)
            
                """
                for _ in range(SIM_TIMES):
                    scene_seed = random.randint(0, 999999999)
                    episode.initialize_episode(scene_seed=scene_seed)
                    for obj in episode.get_objects():
                        if obj['objectType'] not in constants.OBJECTS:
                            continue
                        y, x = game_util.get_object_point(obj, scene_bounds)
                        #print("object %s at " % obj['objectType'], game_util.get_object_point(obj, scene_bounds))
                        obj_id = constants.OBJECT_CLASS_TO_ID[obj['objectType']]
                        memory[y][x][obj_id] += 1
                memory = np.divide(memory, SIM_TIMES)
                """

                print (memory.shape[:2])
                #print (constants.OBJECTS[test_id])
                #plt.figure(figsize=list(memory.shape[:2]))
                #plt.pcolor(memory[:,:,test_id],cmap=plt.get_cmap('Reds'), vmin=0.0, vmax=1.0)#,edgecolors='k', linewidths=1)
                #plt.colorbar()
                #print (memory[:,:,test_id])
                #plt.show()

                fig, axs = plt.subplots(4, 7)
                #a1
                for (i, j) in list(itertools.product("0123","23456")):
                    #print (i, j)
                    obj_id = (int(i))*5+int(j)-1
                    #if obj_id > 20: break
                    ax = axs[int(i), int(j)]
                    pcm = ax.pcolor(memory[:,:,obj_id], cmap=plt.get_cmap('Reds'), vmin=-0, vmax=1)
                    ax.set_title(constants.OBJECTS[obj_id])
                    ax.axis('off')
                
                gs = axs[0, 0].get_gridspec()
                for ax in axs[:,:2].flatten():
                    ax.remove()
                ax_frame = fig.add_subplot(gs[:,:2])
                ax_frame.axis('off')
                
                new_frame = get_agent_map_data(episode.env)
                new_frame = new_frame[:,::-1]
                new_frame = np.rot90(new_frame, -1)
                plt.imshow(new_frame)
                #fig.tight_layout()
                #fig.colorbar(pcm, ax=axs[:])
                fig_path = os.path.join(prefix, str(index)+'.png')
                plt.savefig(fig_path)

                """
                objs = [obj['objectType'] for obj in episode.get_objects()]
                objjs = [obj['objectId'] for obj in episode.get_objects()]
                objjspos = [obj['position'] for obj in episode.get_objects()]
                for ii in range(len(episode.get_objects())):
                    print("object%d at " % ii, game_util.get_object_point(episode.get_objects()[ii], scene_bounds))
                print("objs: ", objs)
                print("objjs: ", objjs)
                print("objjs: ", len(objjs))
                print("objjspos: ", objjspos)
                print("objjspos: ", len(objjspos))
                """
                
                #event = episode.get_env_top_view()
                #print (event.metadata)
                
                #image = drawing.subplot()

            break
            
        #episode.env.stop_unity()

    if DEBUG:
        create_dump()

if __name__ == '__main__':
    main()