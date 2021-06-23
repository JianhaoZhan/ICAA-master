python extract_frames_flows.py '/home/eed-server3/Downloads/dataset/UCF-101/'   '/home/eed-server3/Downloads/dataset/result' 0 101  0



export OPENCV=/usr/local/

g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm





