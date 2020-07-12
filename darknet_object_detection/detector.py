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
                                      py_util.encode(WEIGHT_PATH + 'yolov4-custom_4000.weights'), 0)
        self.meta_custom = dn.load_meta(py_util.encode(WEIGHT_PATH + 'obj_less.data'))
        self.net_origin = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov4.cfg', 'ascii'),
                                      py_util.encode(WEIGHT_PATH + 'yolov4.weights', 'ascii'), 0)
        self.meta_origin = dn.load_meta(py_util.encode(WEIGHT_PATH + 'coco.data', 'ascii'))

        self.count = 0

    def detect(self, image, confidence_threshold=constants.DETECTION_THRESHOLD):
        import darknet as dn
        self.count += 1

        used_inds = []   
        results_custom = dn.detect(self.net_custom, self.meta_custom, image, thresh=confidence_threshold)
        for idx, result in enumerate(results_custom):
            clas, _, box = result
            if box[2] > 200 or box[3] > 200:
                continue
            if py_util.decode(clas) not in constants.OBJECTS_SET:
                continue
            used_inds.append(idx)
        results_custom = [results_custom[i] for i in used_inds]
        #print (results_custom)
        used_inds = []   
        results_origin = dn.detect(self.net_origin, self.meta_origin, image, thresh=confidence_threshold)
        for idx, result in enumerate(results_origin):
            clas, _, _ = result
            if py_util.decode(clas) not in constants.OBJECTS_SET:
                continue
            used_inds.append(idx)
        results_origin = [results_origin[i] for i in used_inds]
        #print (results_origin)
        
        used_inds = list(range(len(results_custom)))
        for idx, result_custom in enumerate(results_custom):
            _, _, box_custom = result_custom
            for result_origin in results_origin:
                _, _, box_origin = result_origin
                lxbound = max(box_custom[0], box_origin[0])
                lybound = max(box_custom[1], box_origin[1])
                
                rxbound = min(box_custom[0]+box_custom[2], box_origin[0]+box_custom[2])
                rybound = min(box_custom[1]+box_custom[3], box_origin[1]+box_custom[3])
                if rxbound < lxbound or rybound < lybound: continue
                inter = (rxbound - lxbound) * (rybound - lybound)
                union = box_custom[2] * box_custom[3] + box_origin[2] * box_origin[3] - inter
                if inter/union > 0.8:
                    used_inds.remove(idx)
                    break
                
        results_custom = [results_custom[i] for i in used_inds]
        #)
        print (results_origin)
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

    if not os.path.exists(DIR_PATH + '/test_images/output'):
        os.mkdir(DIR_PATH + '/test_images/output')

    setup_detectors()
    detector = get_detector()

    t_start = time.time()
    import cv2
    for image_path in TEST_IMAGE_PATHS:
        print('image path: ', image_path)
        image = scipy.misc.imread(image_path)
        (boxes, scores, classes) = detector.detect(image)
        # Visualization of the results of a detection.
        image = visualize_detections(image, boxes, classes, scores)
        scipy.misc.imsave(DIR_PATH + '/test_images/output/' + os.path.basename(image_path), image)
    total_time = time.time() - t_start
    print('total time %.3f' % total_time)
    print('per image time %.3f' % (total_time / len(TEST_IMAGE_PATHS)))