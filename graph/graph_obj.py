import pdb
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import constants

MAX_WEIGHT = 200
EPSILON = 1e-7

# Direction: 0: north, 1: east, 2: south, 3: west


class Graph(object):
    def __init__(self, gt_source_file, use_gt=False, construct_graph=True):
        self.points = (np.load(gt_source_file) * 1.0 / constants.AGENT_STEP_SIZE).astype(int)
        self.xMin = self.points[:, 0].min() - constants.SCENE_PADDING * 2
        self.yMin = self.points[:, 1].min() - constants.SCENE_PADDING * 2
        self.xMax = self.points[:, 0].max() + constants.SCENE_PADDING * 2
        self.yMax = self.points[:, 1].max() + constants.SCENE_PADDING * 2
        gt_edges = {(point[0], point[1]) for point in self.points}
        self.graph = nx.DiGraph()
        self.memory = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.memory75 = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.memory25 = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.empty_memory = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.freq_memory = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.memory[:, :, 0] = 1
        self.construct_graph = construct_graph
        self.index = 0
        self.critical_points = []
        for yy in np.arange(self.yMin, self.yMax + 1):
            for xx in np.arange(self.xMin, self.xMax + 1):
                if use_gt:
                    if (xx, yy) in gt_edges:
                        weight = 1 + EPSILON
                    else:
                        weight = MAX_WEIGHT
                else:
                    weight = 1
                if use_gt:
                    self.memory[int(yy - self.yMin), int(xx - self.xMin), 0] = weight
                if construct_graph:
                    for direction in range(4):
                        node = (xx, yy, direction)
                        back_direction = (direction + 2) % 4
                        back_node = (xx, yy, back_direction)
                        self.graph.add_edge(node, (xx, yy, (direction + 1) % 4), weight=1)
                        self.graph.add_edge(node, (xx, yy, (direction - 1) % 4), weight=1)
                        curr_weight = weight
                        if direction == 0 and yy != self.yMax:
                            self.graph.add_edge((xx, yy + 1, back_direction), back_node, weight=curr_weight)
                        elif direction == 1 and xx != self.xMax:
                            self.graph.add_edge((xx + 1, yy, back_direction), back_node, weight=curr_weight)
                        elif direction == 2 and yy != self.yMin:
                            self.graph.add_edge((xx, yy - 1, back_direction), back_node, weight=curr_weight)
                        elif direction == 3 and xx != self.xMin:
                            self.graph.add_edge((xx - 1, yy, back_direction), back_node, weight=curr_weight)

    def test_graph(self):
        # graph sanity check
        if self.construct_graph:
            for yy in np.arange(self.yMin, self.yMax + 1):
                for xx in np.arange(self.xMin, self.xMax + 1):
                    for direction in range(4):
                        back_direction = (direction + 2) % 4
                        back_node = (xx, yy, back_direction)
                        if direction == 0 and yy != self.yMax:
                            assert(abs(self.graph[(xx, yy + 1, back_direction)][back_node]['weight'] -
                                    self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]) < 0.0001), (
                                    'weight mismatch (%d, %d) %f vs %f' % (xx, yy,
                                            self.graph[(xx, yy + 1, back_direction)][back_node]['weight'],
                                            self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]))

                        elif direction == 1 and xx != self.xMax:
                            assert(abs(self.graph[(xx + 1, yy, back_direction)][back_node]['weight'] -
                                    self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]) < 0.0001), (
                                    'weight mismatch (%d, %d) %f vs %f' % (xx, yy,
                                            self.graph[(xx, yy + 1, back_direction)][back_node]['weight'],
                                            self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]))
                        elif direction == 2 and yy != self.yMin:
                            assert(abs(self.graph[(xx, yy - 1, back_direction)][back_node]['weight'] -
                                    self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]) < 0.0001), (
                                    'weight mismatch (%d, %d) %f vs %f' % (xx, yy,
                                             self.graph[(xx, yy + 1, back_direction)][back_node]['weight'],
                                             self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]))
                        elif direction == 3 and xx != self.xMin:
                            assert(abs(self.graph[(xx - 1, yy, back_direction)][back_node]['weight'] -
                                    self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]) < 0.0001), (
                                    'weight mismatch (%d, %d) %f vs %f' % (xx, yy,
                                            self.graph[(xx, yy + 1, back_direction)][back_node]['weight'],
                                            self.memory[int(yy - self.yMin), int(xx - self.xMin), 0]))
            print('\t\t\tgraph tested successfully')

    def update_graph(self, graph_patch, pose, rows):
        #print("graph update at ", pose)
        graph_patch, curr_val = graph_patch
        curr_val = np.array(curr_val)
        # Rotate the array to get its global coordinate frame orientation.
        if pose[2] != 0:
            graph_patch = np.rot90(graph_patch, pose[2])
        # Shift offsets to global coordinate frame.
        if pose[2] == 0:
            xMin = pose[0] - int(constants.STEPS_AHEAD / 2)
            yMin = pose[1] + 1
        elif pose[2] == 1:
            xMin = pose[0] + 1
            yMin = pose[1] - int(constants.STEPS_AHEAD / 2)
        elif pose[2] == 2:
            xMin = pose[0] - int(constants.STEPS_AHEAD / 2)
            yMin = pose[1] - constants.STEPS_AHEAD
        elif pose[2] == 3:
            xMin = pose[0] - constants.STEPS_AHEAD
            yMin = pose[1] - int(constants.STEPS_AHEAD / 2)
        #print("update rows:", pose[1] - self.yMin, pose[0] - self.xMin)
        #print("update rows:", yMin - self.yMin, yMin + constants.STEPS_AHEAD - self.yMin,
        #            xMin - self.xMin, xMin + constants.STEPS_AHEAD - self.xMin)
        #print("graph update at ", pose)
        if 0 in rows and self.construct_graph:
            for yi, yy in enumerate(range(yMin, yMin + constants.STEPS_AHEAD)):
                for xi, xx in enumerate(range(xMin, xMin + constants.STEPS_AHEAD)):
                    self.update_weight(xx, yy, graph_patch[yi, xi, 0])
            self.memory[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.memory75[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory75[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.memory25[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory25[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.empty_memory[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.empty_memory[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.update_weight(pose[0], pose[1], curr_val[0])
        else:
            self.memory[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.memory75[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory75[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.memory25[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.memory25[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val
            self.empty_memory[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
                    xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = graph_patch
            self.empty_memory[pose[1] - self.yMin, pose[0] - self.xMin, rows] = curr_val

        #self.freq_memory[yMin - self.yMin:yMin + constants.STEPS_AHEAD - self.yMin,
        #                 xMin - self.xMin:xMin + constants.STEPS_AHEAD - self.xMin, rows] = 1.0
        #self.freq_memory[pose[1] - self.yMin, pose[0] - self.xMin, rows] = 1.0
        #print("update freq map")

    def get_graph_patch(self, pose):
        if pose[2] == 0:
            xMin = pose[0] - int(constants.STEPS_AHEAD / 2)
            yMin = pose[1] + 1
        elif pose[2] == 1:
            xMin = pose[0] + 1
            yMin = pose[1] - int(constants.STEPS_AHEAD / 2)
        elif pose[2] == 2:
            xMin = pose[0] - int(constants.STEPS_AHEAD / 2)
            yMin = pose[1] - constants.STEPS_AHEAD
        elif pose[2] == 3:
            xMin = pose[0] - constants.STEPS_AHEAD
            yMin = pose[1] - int(constants.STEPS_AHEAD / 2)
        xMin -= self.xMin
        yMin -= self.yMin

        graph_patch = self.memory[yMin:yMin + constants.STEPS_AHEAD,
                xMin:xMin + constants.STEPS_AHEAD, :].copy()

        if pose[2] != 0:
            graph_patch = np.rot90(graph_patch, -pose[2])

        return graph_patch, self.memory[pose[1] - self.yMin, pose[0] - self.xMin, :].copy()

    def update_weight(self, xx, yy, weight):
        if constants.USE_NAVIGATION_AGENT:
            if self.construct_graph:
                for direction in range(4):
                    node = (xx, yy, direction)
                    self.update_edge(node, weight)
        self.memory[yy - self.yMin, xx - self.xMin, 0] = weight
        self.memory75[yy - self.yMin, xx - self.xMin, 0] = weight
        self.memory25[yy - self.yMin, xx - self.xMin, 0] = weight
        self.empty_memory[yy - self.yMin, xx - self.xMin, 0] = weight

    def update_edge(self, pose, weight):
        (xx, yy, direction) = pose
        back_direction = (direction + 2) % 4
        back_pose = (xx, yy, back_direction)
        if direction == 0 and yy != self.yMax:
            self.graph[(xx, yy + 1, back_direction)][back_pose]['weight'] = weight
        elif direction == 1 and xx != self.xMax:
            self.graph[(xx + 1, yy, back_direction)][back_pose]['weight'] = weight
        elif direction == 2 and yy != self.yMin:
            self.graph[(xx, yy - 1, back_direction)][back_pose]['weight'] = weight
        elif direction == 3 and xx != self.xMin:
            self.graph[(xx - 1, yy, back_direction)][back_pose]['weight'] = weight

    def get_shortest_path(self, pose, goal_pose):
        path = nx.astar_path(self.graph, pose[:3], goal_pose[:3],
                heuristic=lambda nodea, nodeb: (abs(nodea[0] - nodeb[0]) + abs(nodea[1] - nodeb[1])),
                weight='weight')

        # Remove last rotations
        '''
        if not constants.USE_NAVIGATION_AGENT:
            while len(path) > 1:
                if (path[-1][0] == path[-2][0] and
                    path[-1][1] == path[-2][1]):
                    path.pop()
                else:
                    break
        '''

        max_point = 1
        for ii in range(len(path) - 1):
            weight = self.graph[path[ii]][path[ii + 1]]['weight']
            if path[ii][:2] != path[ii + 1][:2]:
                if abs(self.memory[path[ii + 1][1] - self.yMin, path[ii + 1][0] - self.xMin, 0] - weight) > 0.001:
                    print(self.memory[path[ii + 1][1] - self.yMin, path[ii + 1][0] - self.xMin, 0], weight)
                    if constants.USE_NAVIGATION_AGENT:
                        print('nxgraph weights and memory do not match, check that both were updated at all times.')
                    else:
                        print('constants.USE_NAVIGATION_AGENT was False. It may need to be true to get the shortest path.')
                    pdb.set_trace()
            if weight == MAX_WEIGHT:
                break
            max_point += 1
        path = path[:max_point]

        actions = [self.get_plan_move(path[ii], path[ii + 1]) for ii in range(len(path) - 1)]
        return actions, path

    def get_plan_move(self, pose0, pose1):
        if (pose0[2] + 1) % 4 == pose1[2]:
            action = {'action': 'RotateRight'}
        elif (pose0[2] - 1) % 4 == pose1[2]:
            action = {'action': 'RotateLeft'}
        else:
            action = {'action': 'MoveAhead', 'moveMagnitude' : constants.AGENT_STEP_SIZE}
        return action

    def get_shifted_pose(self, pose):
        new_pose = np.array(pose)
        new_pose[0] -= self.xMin
        new_pose[1] -= self.yMin
        return tuple(new_pose.tolist())

    def memory_decay(self, name):
        print ("memory decay\n")
        #calculate criticals
        """
        self.critical_points = []
        for obj_id in range(1, 21):
            sil = []
        
            mask = np.array(np.where(self.memory[:,:,obj_id] > 0)).transpose()
            
            if mask.shape[0] == 0:
                self.critical_points.append(np.array([], dtype="int"))
                continue
            kmax = min(10, mask.shape[0]-1)

            for k in range(1, kmax):
                kmeans = KMeans(n_clusters = k).fit(mask)
                labels = kmeans.labels_
                #print(k)
                #print(np_map[:,:,obj_id].shape)
                #print(labels)
                if np.unique(labels).shape[0] == 1:
                    sil.append(0)
                    continue
                sil.append(silhouette_score(mask, labels, metric = 'euclidean')) 
            if len(sil) == 0:
                self.critical_points.append(np.array([], dtype="int"))
                continue
            k = sil.index(max(sil))+2
            #print(k)
            kmeans = KMeans(n_clusters = k).fit(mask)
            centers = kmeans.cluster_centers_
            self.critical_points.append(np.array(centers, dtype="int"))
            #print(centers)
        #for i in range(len(self.critical_points)):
        #    print(i, ": ", self.critical_points[i])
        """


        
        #print (self.memory)
        #mask = np.where(self.memory < 1)
        mask25 = np.where(self.memory25 < 1)
        mask75 = np.where(self.memory75 < 1)
        empty_mask = np.where(self.empty_memory > 0)
        #print(self.freq_memory)
        #np.save('%d_%s_memory.npy' % (self.index, name), self.memory)
        #np.save('%d_%s_memory75.npy' % (self.index, name), self.memory75)
        #np.save('%d_%s_memory25.npy' % (self.index, name), self.memory25)
        #np.save('%d_%s_empty_memory.npy' % (self.index, name), self.empty_memory)
        #np.save('%d_freq_memory.npy' % self.index, self.freq_memory)
        
        #self.memory[:, :, 1:] = self.memory[:, :, 1:] * constants.MAP_FACTOR
        self.memory[mask] = self.memory[mask] * constants.MAP_FACTOR
        self.memory75[mask75] = self.memory75[mask75] * 0.75
        self.memory25[mask25] = self.memory25[mask25] * 0.25
        #self.memory = self.memory * constants.MAP_FACTOR
        self.empty_memory = np.zeros((self.yMax - self.yMin + 1, self.xMax - self.xMin + 1, 1 + constants.NUM_CLASSES), dtype=np.float32)
        self.freq_memory = self.freq_memory * constants.FREQ_TH
        self.freq_memory[empty_mask] += 0.3
        self.freq_memory[self.freq_memory>1.0] = 1.0
        #np.save('%d_freq_memory.npy' % self.index, self.freq_memory)
        
        print ("current task index: %d decaying" % self.index)
        self.index = self.index +1
