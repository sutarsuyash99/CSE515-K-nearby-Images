import utils

class Task10:
    def __init__(self) -> None:
        pass

    def runTask10(self) -> None:
        # take label input
        # take semantic label input
        # take user input of k
        # display top k image
        label_selected = utils.get_user_input_label()
        pathname, option = utils.get_user_input_latent_semantics()
        k = utils.get_user_input_k()
        print(pathname, option, k, label_selected)


if __name__ == '__main__':
    task10 = Task10()
    task10.runTask10()