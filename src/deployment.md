# Anaconda Deployment Instructions

To install the necessary environment for you to run the application, please follow the following steps:

## Steps
Note: Only run this once to have the environment setup.
0. Have Anaconda Installed with Administrator permissions
1. Go to this directory of this repository.
2. To activate and build environment, run:
```
conda env create -f environment.yml
```

## Testing
Test to see if application works for you by running the following steps:
1. cd into `/application`
2. In application directory, run:
```
flask run
```
3. Open up the link: `http://127.0.0.1:5000/`, to view the web application.
4. Test and see if you have two camera windows:
    * One for original camera feed
    * The other a processed video stream of open pose skeleton and finger render.
    * And that your camera light is ON, on your laptop
