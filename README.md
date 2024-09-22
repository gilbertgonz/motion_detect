# Simple motion detection
Detecting moving objects using frame subtraction. It computes differences between consecutive frames. For noise reduction, two methods were used:
- Non-maximal suppression to filter overlapping bounding boxes
- Dilation for filling in gaps and connecting nearby contours

Some results:

https://github.com/user-attachments/assets/458064b9-d67c-4822-90ba-88468b657fe2

Data taken from here: https://medium.com/@itberrios6/introduction-to-motion-detection-part-1-e031b0bb9bb2

https://github.com/user-attachments/assets/d36c9615-eb88-4a3e-b339-64ff23264b9a

I recorded this on my way home from work, thought this would be a fun test case since the object in motion is so small (it was a small aircraft). Luckily, the clear Bay Area skies made it easy to work with :)

## To run:
1. Install [docker](https://docs.docker.com/engine/install/)

2. Clone repo

3. Build:
```
$ docker build -t motion_detect .
```

4. Run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix motion_detect
```
