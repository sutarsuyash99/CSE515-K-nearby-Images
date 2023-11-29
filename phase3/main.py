from utils import int_input, initialise_project
from Task0a import Task0a
from Task0b import Task0b
from Task1 import Task1
from Task2 import Task2
from Task3 import Task3
from Task4 import Task4a
from Task4 import Task4b
from Task5 import Task5
import os
import json



def task0_main():
    print("="*25, 'SUB-MENU', '='*25) 
    print('\n\nSelect your sub-task:')
    print("1. Task 0a\n2. Task 0b\n")
    option = int_input(1)
    if option == 1:
        task0a = Task0a()
        task0a.all_data()
    else:
        if option != 2:
            print("Invalid option. Running with default value - 2")
        task0b = Task0b()
        task0b.all_data()

def task1_main():
    task1 = Task1()
    task1.runTask1(3)

def task2_main():
    task1 = Task2()
    task1.execute()

def task3_main():
    task3 = Task3()
    task3.run_classifiers()

# /Users/suyashsutar99/Downloads/ImportantDoc.jpg
def task4_main():
    print("="*25, 'SUB-MENU', '='*25) 
    print('\n\nSelect your sub-task:')
    print("1. Task 4a\n2. Task 4b\n")
    option = int_input(1)
    if option == 2:
        task4b = Task4b()
        task4b.runTask4b()
    else:
        if option != 1:
            print("Invalid option. Running with default value - 1")
        task4a = Task4a()
        task4a.runTask4a()

def task5_main():
    file_path = '4b_output.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            task_4b = json.load(file)
    else:
        print(f"Please Run Task 4B before Task 5")

    query_vector = []
    task5 = Task5(task_4b)
    while True:
        print("Running Task 5")
        new_vector = task5.run_feedback(query_vector)
        task4 = Task4b()
        query_vector = task4.runTask4b(imageID = task_4b["query_image"], query_vector = new_vector)
        exit_condition = input("To exit Relevance Feedback press 0 to Give further feedback press 1 -  ")
        if exit_condition.lower() == '0':
            break


option = -1
while option != 99:
    print("\n\n"+"-"*25, 'MENU', '-'*25)
    print('Select your option:\
        \n\n\
        \n0. Task 0\
        \n1. Task 1\
        \n2. Task 2\
        \n3. Task 3\
        \n4. Task 4\
        \n5. Task 5\
        \n99. Quit\
        \n\n')
    option = int_input()
    # I made a typo to name functions like where query should be 'label'
    # too lazy to correct
    match option:
        case 99: print('Exiting...')
        case 0: task0_main()
        case 1: task1_main()
        case 2: task2_main()
        case 3: task3_main()
        case 4: task4_main()
        case 5: task5_main()
    
        case default: print('No matching input was found')