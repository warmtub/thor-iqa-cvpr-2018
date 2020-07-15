import glob
import numpy as np
import scipy.misc
import os
import time
import constants

import threading

from utils import bb_util
from utils import drawing
from utils import py_util

import rospy
import ros_numpy
from sensor_msgs.msg import Image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = os.path.join(DIR_PATH, 'yolo_weights/')


class ObjectDetector(object):
    def __init__(self, detector_num=0):
        import darknet as dn
        dn.set_gpu(int(constants.DARKNET_GPU))
        self.detector_num = detector_num
        #self.net = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov3-thor.cfg'),
        #                       py_util.encode(WEIGHT_PATH + 'yolov3-thor_final.weights'), 0)
        #self.meta = dn.load_meta(py_util.encode(WEIGHT_PATH + 'thor.data'))
        self.net_custom = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov4-custom.cfg'),
                                      py_util.encode(WEIGHT_PATH + 'yolov4-custom_2000.weights'), 0)
        self.meta_custom = dn.load_meta(py_util.encode(WEIGHT_PATH + 'obj_less.data'))
        self.net_origin = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov4.cfg', 'ascii'),
                                      py_util.encode(WEIGHT_PATH + 'yolov4.weights', 'ascii'), 0)
        self.meta_origin = dn.load_meta(py_util.encode(WEIGHT_PATH + 'coco.data', 'ascii'))

        self.count = 0

    def detect(self, image, confidence_threshold=constants.DETECTION_THRESHOLD):
        import darknet as dn
        self.count += 1

        used_inds = []
        results_custom = dn.detect(self.net_custom, self.meta_custom, image, thresh=0.2)
        image[:,:,[0, 2]] = image[:,:,[2, 0]]
        for idx, result in enumerate(results_custom):
            clas, score, box = result
            if box[2] > 150 or box[3] > 100:
                continue
            if py_util.decode(clas) not in constants.OBJECTS_SET:
                continue
            used = True
            for jdx in range(0, len(results_custom)):
                clasj, scorej, boxj = results_custom[jdx]
                #print (clas, ' v s ', clasj)
                if clas == clasj and score < scorej: used = False
                lxbound = max(box[0], boxj[0])
                lybound = max(box[1], boxj[1])
                
                rxbound = min(box[0]+box[2], boxj[0]+boxj[2])
                rybound = min(box[1]+box[3], boxj[1]+boxj[3])
                if rxbound < lxbound or rybound < lybound: continue
                inter = (rxbound - lxbound) * (rybound - lybound)
                union = box[2] * box[3] + boxj[2] * boxj[3] - inter
                #print (score , scorej, ' with ', inter/union)
                if inter/union > 0.6 and score < scorej: used = False
            #print (used)
            if used: used_inds.append(idx)
        results_custom = [results_custom[i] for i in used_inds]
        #print (results_custom)
        used_inds = []   
        results_origin = dn.detect(self.net_origin, self.meta_origin, image, thresh=0.5)
        for idx, result in enumerate(results_origin):
            clas, score, _ = result
            if py_util.decode(clas) not in constants.OBJECTS_SET:
                continue
            used = True
            for jdx in range(0, len(results_custom)):
                clasj, scorej, boxj = results_custom[jdx]
                #print (clas, ' v s ', clasj)
                if clas == clasj and score < scorej: used = False
                lxbound = max(box[0], boxj[0])
                lybound = max(box[1], boxj[1])
                
                rxbound = min(box[0]+box[2], boxj[0]+boxj[2])
                rybound = min(box[1]+box[3], boxj[1]+boxj[3])
                if rxbound < lxbound or rybound < lybound: continue
                inter = (rxbound - lxbound) * (rybound - lybound)
                union = box[2] * box[3] + boxj[2] * boxj[3] - inter
                #print (score , scorej, ' with ', inter/union)
                if inter/union > 0.6 and score < scorej: used = False
            #print (used)
            if used: used_inds.append(idx)
        results_origin = [results_origin[i] for i in used_inds]
        #print (results_origin)
        
        used_inds = list(range(len(results_custom)))
        for idx, result_custom in enumerate(results_custom):
            _, _, box_custom = result_custom
            for result_origin in results_origin:
                _, _, box_origin = result_origin
                lxbound = max(box_custom[0], box_origin[0])
                lybound = max(box_custom[1], box_origin[1])
                
                rxbound = min(box_custom[0]+box_custom[2], box_origin[0]+box_origin[2])
                rybound = min(box_custom[1]+box_custom[3], box_origin[1]+box_origin[3])
                if rxbound < lxbound or rybound < lybound: continue
                inter = (rxbound - lxbound) * (rybound - lybound)
                union = box_custom[2] * box_custom[3] + box_origin[2] * box_origin[3] - inter
                if inter/union > 0.7:
                    used_inds.remove(idx)
                    break
                
        results_custom = [results_custom[i] for i in used_inds]
        #)
        #print (results_origin)
        results = results_custom + results_origin
        print (results)
        #print(union)
        #print(inter)

        
        if len(results) > 0:
            classes, scores, boxes = zip(*results)
        else:
            classes = []
            scores = []
            boxes = np.zeros((0, 4))
        boxes = np.array(boxes)
        boxes = bb_util.xywh_to_xyxy(boxes.T).T
        scores = np.array(scores)
        classes = np.array([py_util.decode(cls) for cls in classes])
        return boxes, scores, classes

def visualize_detections(image, boxes, classes, scores):
    out_image = image.copy()
    #if len(boxes) > 0:
    #    boxes = (boxes / np.array([constants.SCREEN_HEIGHT * 1.0 / image.shape[1],
    #            constants.SCREEN_WIDTH * 1.0 / image.shape[0]])[[0, 1, 0, 1]]).astype(np.int32)
    #print (boxes)
    for ii,box in enumerate(boxes):
        drawing.draw_detection_box(out_image, box, classes[ii], confidence=scores[ii], width=2)
    return out_image


singleton_detector = None
detectors = []
def setup_detectors(num_detectors=1):
    global detectors
    for dd in range(num_detectors):
        detectors.append(ObjectDetector(dd))

detector_ind = 0
detector_lock = threading.Lock()
def get_detector():
    global detectors, detector_ind
    detector_lock.acquire()
    detector = detectors[detector_ind % len(detectors)]
    detector_ind += 1
    detector_lock.release()
    return detector

image = 0
detector = 0
def image_cb(data):
    global image, detector
    image = ros_numpy.numpify(data)
    (boxes, scores, classes) = detector.detect(image)
    print (classes)


if __name__ == '__main__':
    """
    global detector
    rospy.init_node('game_state', anonymous=True)
    setup_detectors()
    detector = get_detector()
    image_sub = rospy.Subscriber("/eyecam/color/image_raw", Image, image_cb)
    rospy.spin()
        

    """
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = DIR_PATH + '/test_images'
    TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.JPG')))
    #TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.npy')))

    if not os.path.exists(DIR_PATH + '/test_images/output'):
        os.mkdir(DIR_PATH + '/test_images/output')

    setup_detectors()
    detector = get_detector()

    t_start = time.time()
    import cv2
    for image_path in TEST_IMAGE_PATHS:
        print('image path: ', image_path)
        #image = scipy.misc.imread(image_path)
        image = cv2.imread(image_path)
        #image = np.load(image_path)
        (boxes, scores, classes) = detector.detect(image)
        # Visualization of the results of a detection.
        image = visualize_detections(image, boxes, classes, scores)
        scipy.misc.imsave(DIR_PATH + '/test_images/output/' + os.path.basename(image_path) + '.jpg', image)
    total_time = time.time() - t_start
    print('total time %.3f' % total_time)
    print('per image time %.3f' % (total_time / len(TEST_IMAGE_PATHS)))