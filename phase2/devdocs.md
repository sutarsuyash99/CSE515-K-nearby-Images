## Installation and venv setup

1. Run `startup.sh` or `startup.bat` for setting up venv on linux/unix or windows respectively.
2. If you don't find a dependency add it and update `requirements.txt` (see: https://stackoverflow.com/questions/31684375/automatically-create-file-requirements-txt)

### Folder structure:

1. `main.py` will be our entry point
2. `util.py` and `common.py` will be shared codebase for utility function
3. `data` folder will contain the database
4. You can use `test.py` (It is not updated or versioned on git)

### Mongo Installation Steps
brew install mongodb-community
brew services start mongodb/brew/mongodb-community "To activate the database"
mongorestore --db feature_descriptors /Users/hardikpatel/Desktop/CSE515-K-nearby-Images/feature_descriptors if you have the data

