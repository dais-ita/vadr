# Visual & Acoustic Discriminative Relevance demo
[![Conference](http://img.shields.io/badge/DAIS--ITA_AFM-2019-blue.svg?style=flat-square)](https://dais-ita.org/node/4034)

## Gesture Detection Model
...


## Speech Commands Model
...
## UI/API instructions

        cd ui/
        npm update
        npm start

open browser (Chrome) on http://localhost:3000/
Drop down the humburger icon and select 'Reset test data'
insights should start to appear every few seconds and you can click around the user interface to explore the results
To reset the environment simply choose 'Reset test data' from the hamburger menu at any time.

There is also docs (a video) in the docs/ and docs/video/ folder that shows a quick demo

## Flask services
cd flask/video/webcam
python main.py
This will start a stream of your webcam on localhost:5001 and the live feed itself is on localhost:5001/video_feed (which is used by the UI as the stream source)

### Installation instructions
1. Get the modified `audtorch` library and install it in develop mode 
(until PR is accepted)
        
        git clone https://github.com/harritaylor/audtorch ~/path/to/repos/
        pip install -e ~/path/to/repos/audtorch # install audtorch in dev mode
        
    > If this fails, install the sox library if not already installed.  
2. Install requirements (preferably in a venv)
    
        pip install -r requirements.txt
