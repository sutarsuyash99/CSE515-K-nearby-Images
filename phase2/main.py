from utils import int_input, initialise_project
from Task0a import Task0a
from Task0b import Task0b
from Task1 import Task1
from Task2a import Task2a
from Task2b import Task2b
from Task3 import task3
from Task4 import task4
from Task5 import task5
from Task6 import task6
from Task7 import task7
from Task8 import Task8
from Task9 import task9
from Task10 import Task10
from Task11 import Task11


def task0_main():
    print("="*25, 'SUB-MENU', '='*25) 
    print('\n\nSelect your sub-task:')
    print("1. Task 0a\n2. Task 0b\n")
    option = int_input(1)
    if option == 1:
        task0a = Task0a()
        task0a.runTask0a()
    else:
        if option != 2:
            print("Invalid option. Running with default value - 2")
        task0b = Task0b()
        task0b.image_image_distance()

def task1_main():
    task1 = Task1()
    task1.query_image_top_k()

def task2_main():
    print("="*25, 'SUB-MENU', '='*25)
    print('\n\nSelect your sub-task:')
    print("1. Task 2a\n2. Task 2b\n")
    option = int_input(1)
    if option == 1:
        task2a = Task2a()
        task2a.image_query_top_k()
    else:
        if option != 2:
            print("Invalid option. Running with default value - 2")
        task2b = Task2b()
        task2b.resnet_50_image_label_topk()

def task3_main():
    Task3 = task3()
    Task3.k_latent_semantics()

# /Users/suyashsutar99/Downloads/ImportantDoc.jpg
def task4_main():
    Task4 = task4()
    Task4.LS2_cp_decompose()

def task5_main():
    Task5 = task5()
    Task5.runTask5()

def task6_main():
    Task6 = task6()
    Task6.image_image_ls()

def task7_main():
    Task7 = task7()
    Task7.image_in_image_out()

def task8_main():
    task8 = Task8()
    task8.runTask8()

def task9_main():
    Task9 = task9()
    Task9.menu()

def task10_main():
    task10 = Task10()
    task10.runTask10()

def task11_main():
    task11 = Task11()
    task11.runTask11()


option = -1
while option != 99:
    print("-"*25, 'MENU', '-'*25)
    print('Select your option:\
        \n\n\
        \n0. Task 0\
        \n1. Task 1\
        \n2. Task 2\
        \n3. Task 3\
        \n4. Task 4\
        \n5. Task 5\
        \n6. Task 6\
        \n7. Task 7\
        \n8. Task 8\
        \n9. Task 9\
        \n10. Task 10\
        \n11. Task 11\
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
        case 6: task6_main()
        case 7: task7_main()
        case 8: task8_main()
        case 9: task9_main()
        case 10: task10_main()
        case 11: task11_main()
    
        case default: print('No matching input was found')