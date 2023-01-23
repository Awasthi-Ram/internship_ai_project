import cv2
import numpy as np
import datetime
cap = cv2.VideoCapture("")
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(int(1000/fps))

    if key == ord("q"):
        break

def detect_ball_color(frame):
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_range = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    green_range = cv2.inRange(hsv, (35, 100, 100), (75, 255, 255))
    blue_range = cv2.inRange(hsv, (110, 100, 100), (130, 255, 255))

    red_mask = cv2.bitwise_and(frame, frame, mask=red_range)
    green_mask = cv2.bitwise_and(frame, frame, mask=green_range)
    blue_mask = cv2.bitwise_and(frame, frame, mask=blue_range)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    if red_pixels > green_pixels and red_pixels > blue_pixels:
        return 'red'
    elif green_pixels > red_pixels and green_pixels > blue_pixels:
        return 'green'
    elif blue_pixels > red_pixels and blue_pixels > green_pixels:
        return 'blue'
    else:
        return 'unknown'


def detect_quadrants(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quadrants = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            quadrants.append((approx, (cX, cY)))

    return quadrants
prev_frame = None
current_frame = None
ret, current_frame = cap.read()

while ret:

    prev_frame = current_frame.copy()
    ret, current_frame = cap.read()



def track_balls(prev_frame, current_frame, quadrants, min_distance=10):
    prev_frame = prev_frame.copy()
    prev_balls = detect_ball_color(prev_frame)
    current_balls = detect_ball_color(current_frame)
    tracks = []
    events = []
    for prev_ball in prev_balls:
        closest_ball = None
        min_dist = np.inf
        for current_ball in current_balls:
            dist = np.linalg.norm(prev_ball[1]-current_ball[1])
            if dist < min_dist:
                closest_ball = current_ball
                min_dist = dist
        if min_dist < min_distance:
            tracks.append((prev_ball[0], closest_ball[0]))
            for quadrant in quadrants:
                prev_in_quadrant = cv2.pointPolygonTest(quadrant[0], prev_ball[1], False) >= 0
                current_in_quadrant = cv2.pointPolygonTest(quadrant[0], closest_ball[1], False) >= 0

                if prev_in_quadrant != current_in_quadrant:
                    event_type = "entry" if current_in_quadrant else "exit"
                    event = (closest_ball[0], quadrant[1], event_type)
                    events.append(event)
                    
    return tracks, events




def record_events(events, video_path, output_file):
    with open(output_file, 'w') as f:
        f.write("Time, Quadrant Number, Ball Colour, Type (Entry or Exit)\n")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for event in events:
            frame_num = event[0]
            timestamp = frame_num / fps
            quadrant_num = event[1]
            ball_color = event[2]
            event_type = event[3]
            f.write("{}, {}, {}, {}\n".format(timestamp, quadrant_num, ball_color, event_type))
    cap.release()


def display_video(video_path, tracks, events):
    cap1 = cv2.VideoCapture(video_path)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    timestamp = 0


    while True:

        ret, frame = cap1.read()
        if not ret:
            break
        for track in tracks:
            if frame_num >= track[0] and frame_num <= track[1]:
                cv2.line(frame, track[2], track[3], (0, 0, 255), 2)
       # for event in events:
        
        cap1.release()
print(detect_ball_color(frame))
track_balls(prev_frame,current_frame,detect_quadrants(frame))
record_events(track_balls(prev_frame,current_frame,detect_quadrants(frame))[1],"project.mp4","record.csv")     
display_video(project.mp4,track_balls(prev_frame,current_frame,detect_quadrants(frame))[0],track_balls(prev_frame,current_frame,detect_quadrants(frame))[1])
cap.release()
cv2.destroyAllWindows()
