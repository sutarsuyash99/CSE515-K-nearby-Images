## Phase 3 implementation:

***INFO:*** Dataset used here is `Caltech_101` from `torchvision`

***NOTE:*** This readme delves more into second phase of project

### Pre-installation requirements:

1. `python3` installed and working (Recommended version 3.11.4 -- this is the version on author's machine)
2. First time installation on a new machine requires Internet connection -- downloading libraries and artifacts
3. Have MongoDB installed and running (read: https://www.mongodb.com/docs/manual/installation/)
4. Download MongoRepo from here: https://arizonastateu-my.sharepoint.com/:f:/g/personal/pbarbhay_sundevils_asu_edu/ErKQmQ6owOZGr7qtHbTFZC8Bab1CIGlFfBys756mDNYjkQ?e=MJbXhU
5. Run the command: 
```
mongorestore --db feature_descriptors <path_to_downloaded_content>
```
6. To successfully view and manage data, you may opt to download mongoDB compass

### Configuration Steps:

- For Unix/Linux systems:
    1. Run `. startup.sh` in the subfolder containing that file 
    (Sometimes `bash startup.sh` works better) 
- For Windows systems:
    1. Run `startup.bat` in the subfolder containing that file

### How to run programs:

The project is split into multiple subtasks, to run the individual tasks, use the command:
```
python3 Task<individual_file>.py
```

For example: To run Task 1 files, use the command:
```
python3 Task1.py
```

***NOTE:*** For any tasks with subtasks like 0 and 2, run the 0a/0b or 2a/2b respectively.

To run the file mulitple times, use 
```
python3 main.py
```