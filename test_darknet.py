import os
import darknet as dn
from utils import py_util
from PIL import Image
from numpy import asarray
import rospy
import ros_numpy
from sensor_msgs.msg import Image as ImageTopic


image_ros = 0
def image_cb(data):
    global image_ros
    print ("image callback triggered")
    image_ros = ros_numpy.numpify(data)#[:,80:560]
    #print(self.image.shape)

rospy.init_node('pls_darknet_pls', anonymous=True)
image_sub = rospy.Subscriber("/camera/color/image_raw", ImageTopic, image_cb)
#rospy.spin()

cwd = os.getcwd()
WEIGHT_PATH = os.path.join(cwd, 'darknet_object_detection' ,'yolo_weights/')
net = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov4.cfg', 'ascii'),
                  py_util.encode(WEIGHT_PATH + 'yolov4.weights', 'ascii'), 0)
net = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov4-custom.cfg', 'ascii'), py_util.encode(WEIGHT_PATH + 'yolov4-custom_4000.weights', 'ascii'), 0)
meta = dn.load_meta(py_util.encode(WEIGHT_PATH + 'coco.data', 'ascii'))
meta = dn.load_meta(py_util.encode(WEIGHT_PATH + 'obj_less.data', 'ascii'))
image = Image.open('1593506518119233388.jpeg')
data = asarray(image)
print(data)
results = dn.detect(net, meta, data, thresh=0.2)
#results = dn.detect(net, meta, image_ros, thresh=0.2)
#results = dn.performDetect('1593504186807737616.jpeg')
print(results)