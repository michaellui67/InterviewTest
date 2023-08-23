import numpy as np
import cv2
from keras.preprocessing import image
import time
import threading
from greenlet import getcurrent as get_ident

class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # start background frame thread
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait until first frame is available
            BaseCamera.event.wait()

    
    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame
                        
    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None

class Camera(BaseCamera):
    @staticmethod
    def frames():
        face_cascade = cv2.CascadeClassifier('EmosionDetector/haarcascade_frontalface_default.xml')
        
        cap = cv2.VideoCapture(0)
		#-----------------------------
		#face expression recognizer initialization
        from keras.models import model_from_json
        model = model_from_json(open("EmosionDetector/facial_expression_model_structure.json", "r").read())
        model.load_weights('EmosionDetector/facial_expression_model_weights.h5') #load weights

		#-----------------------------
        
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        
        while(True):
            ret, img = cap.read()
			#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

			#print(faces) #locations of detected faces

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
				
                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
				
                img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
				
                predictions = model.predict(img_pixels) #store probabilities of 7 expressions
				
				#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                max_index = np.argmax(predictions[0])
				
                emotion = emotions[max_index]
				
				#write emotion text above rectangle
                cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
				
				#process on detected face end
				#-------------------------

            yield cv2.imencode('.jpg',img)[1].tobytes()