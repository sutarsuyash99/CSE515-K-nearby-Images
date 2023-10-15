import utils
from label_vectors import create_labelled_feature_vectors
import distances
from Mongo.mongo_query_np import get_all_feature_descriptor
import torch
from ppagerank import pagerank
from label_vectors import get_all_label_feature_vectors
import numpy as np
from topk import query_label_image_top_k, query_label_image_top_k_ls

class Task11:

    def __init__(self):
        self.dataset, self.labelled_images = utils.initialise_project()


    def get_label(self) :
    
        #Get the label id 
        label_index_selected = utils.get_user_input_label()
        try :
            label_selected = self.dataset.categories[label_index_selected]
        except Exception :
            print("label not in database -> try again")
            return None, None
            
        print(f"Input provided: {label_index_selected} => {label_selected}")
        return label_index_selected, label_selected
        
    def get_label_representatives(self, feature_model : str , feature : int, label_name :str,label_id : int, latent_space : int = None, latent_semantic : str = None, labelled_images : dict = None ) -> list :
        
        # For a feature space or latent space and feature space get label representatives
        # 2 cases 
        print("\nHow many seed images to use for a label : default 1 : ")
        k = input("Press enter for default value    ")
        
        if not k :
            k = 1
        else :
            k = int(k)
        print('#### Seed used ####: ')
        
        #Case 1 : Only feature model 
        if latent_space == None :
            seeds = query_label_image_top_k(k, feature_model, feature, label_name, label_id)
        
        #Case 2 : Latent Space and feature model
        else :
            seeds = query_label_image_top_k_ls(k, feature_model, feature, label_name, label_id, latent_space, latent_semantic, labelled_images)
        print('###################: ')    
        return seeds
        
    
    def pagerankcall(self, matrix, seeds : list, m : int, n : int, label : str):
    
        rankings = pagerank(matrix, seeds, m, n) 
                    
        #Covert ranking ids to even ids
        even_rankings = [(image_id * 2, score) for image_id , score in rankings]
                    
        #Display rankings 
        print(f'Pagerank score for label {label} : ')
        print(even_rankings)
        utils.display_k_images_subplots(
            self.dataset, even_rankings, f"Top {m} images for {label}"
        )
    
    def runTask11(self):
    
        print("=" * 25, "SUB-MENU", "=" * 25)
        
        #Get the feature space or latent space with feature space to operate
        option = utils.get_user_input_model_or_space()


        match option :
            
            case 1 : 
            
                '''
                Case 1: Only operate in feature space 
                '''
                
                #Get feature model 
                print("\n")
                print("*"*25 + " Feature Model "+ "*"*25)
                print("Please select from below mentioned options")
                model_space, feature, dbName = utils.get_user_selected_feature_model()
                
                feature_model = utils.feature_model[feature]
                
                
                #check if the image-image similarity matrix exist if not ask user to run the task 6
                file = utils.get_saved_model_files(feature_model=feature_model)
                if file == None :
                    print(f'No saved image-image similarity model for {feature_model}')
                    print(f'Creating image image similarity score model on demand....')
                    matrix = utils.generate_image_similarity_matrix_from_db(feature_model, feature)
                else :
                
                    matrix = torch.load(file)
                    print("Model loaded...\n")
                    
                    
                #Get label id
                while True :
                    label_id, _ = self.get_label()
                    if label_id != None :
                        break
                label_name = self.dataset.categories[label_id]
                
                #Get label representative in particular feature model
                seeds = self.get_label_representatives( feature_model, feature, label_name, label_id)
                
                print("\nEnter value for number of similar images to find - m : ")
                m = utils.int_input()
                
                print("\nEnter value for n : ")
                n = utils.int_input()
                
                self.pagerankcall(matrix, seeds, m, n, label_name)
            
            
            case 2 : 
                
                '''
                Case 2: Operate in latent space and feature model
                '''
                
                #Get user selected latent space and corresponding feature space and dimensionality reduction 
                print("\n")
                print("*"*25 + " Latent Space "+ "*"*25)
                print("Please select from below mentioned options")
                
                #latent space, feature space and dimensionality reduction/letant semantics 
                ls_option, fs_option, dr_option = utils.get_user_selected_latent_space_feature_model()
                
                #special case for LS2 - CP decomposition for others :
                if ls_option != 2 :
                    d_reduction = utils.latent_semantics[dr_option]
                else :
                    d_reduction = dr_option

                #feature_model name
                feature_model = utils.feature_model[fs_option]
                
                #Check if model exists for that options 
                file = utils.get_saved_model_files(feature_model=feature_model, latent_space=ls_option, d_reduction=d_reduction)
                if file == None :
                    print(f'No saved model for LS{ls_option} - {feature_model} - {d_reduction}')
                    print(f'Please run task 3-6 to generate model for LS{ls_option} - {feature_model} - {d_reduction}')
                else :
                    f_name = file.split('\\')[-1]
                    
                    print(f"Factors file exist for the selected option - {f_name} feature")
                    file = torch.load(file)
                    
                    #Special case for LS2 : image weight pairs 
                    if ls_option == 2 :
                        file = file[1][0]

                    print("File loaded...\n")
                    
                    #Special case for LS5  : label_label matrix 
                    if ls_option == 3 :
                            
                        print("Generating image_image similarity matrix by mapping the label-label matrix.....")
                        matrix = utils.generate_matrix_from_label_label_matrix(file, fs_option, self.labelled_images)
                        
                    else :    
                        #check if image_image matrix file exits
                        #if not then generate 
                        print("Generating scores based on the weights....")
                        matrix = utils.generate_matrix_from_image_weight_pairs(file, fs_option)
                        print("Model loaded...\n")
                    
                    while True :
                        label_id, _ = self.get_label()
                        if label_id != None :
                            break
                    label_name = self.dataset.categories[label_id]
                
                    #Get label representative in particular feature model
                    seeds = self.get_label_representatives( feature_model, fs_option, label_name, label_id, ls_option, d_reduction, self.labelled_images )
                    
                    print("\nEnter value for number of similar images to find - m : ")
                    m = utils.int_input()
                
                    print("\nEnter value for n : ")
                    n = utils.int_input()
                    
                    self.pagerankcall(matrix, seeds, m, n, label_name)
                

if __name__ == "__main__":
    task11 = Task11()
    task11.runTask11()