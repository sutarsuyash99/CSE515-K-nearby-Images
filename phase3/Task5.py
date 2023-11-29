import numpy as np

import utils
import Mongo.mongo_query_np as mongo_query
import classifiers
import svm_feedback as svm_feedback
import dimension_reduction
import os
import json
import Task4


class Task5:
    def __init__(self, task4b_output):
        self.dataset, self.labelled_images = utils.initialise_project()
        self.task_4b = task4b_output

    def print_labels(self, result: list) -> None:
        print("-" * 40)
        for i in result:
            print(i)
    def get_user_feedback(self, task_4b= None):
        """Gets users feedback from command line"""
        feedback_map = {}
        print(" \n Please enter Image Id you want to give feedback on")

        print("Type feedback as - Very Relevent - +R, Relevent - R, Irelevent - I, Very Irelevent - +I \n")
        while True:
            # Get image ID from the user
            image_id = input("Enter Image IDs Commas Seprated (or type 'e' to exit): ")

            image_list = image_id.split(",")
            if image_id.lower() == 'e':
                break

            feedback = input("Enter Feedback: ")

            # Store the feedback in the dictionary with image ID as the key
            for val in image_list:
                feedback_map[int(val)//2] = feedback

        # Display the feedback map
        print("Feedback Map:")
        for image_id, feedback in feedback_map.items():
            print(f"Image ID: {image_id*2}, Feedback: {feedback}")
        return feedback_map
    
    def probaility_feedback(self, feedback, image_vectors, query_vector):
        """Gets feedback from users and makes changes to the query accoridingly"""
        # alpha=1
        # beta= 0.5
        # rel_img = [image_vectors[x] for x, y in feedback.items() if y == "+R"]
        # rel_centroid = np.mean(rel_img, axis=0)
    
        # # Compute the centroid of the non-relevant documents
        # irel_img = [image_vectors[x] for x, y in feedback.items() if y == "+I"]
        # non_rel_centroid = np.mean(irel_img, axis=0)
        # print(non_rel_centroid)
        
        # # Compute the new query vector
        # new_query = alpha * query_vector + beta * rel_centroid - 0.5 * non_rel_centroid
        # return new_query
        beta = 0.1
        image_mean = np.mean(image_vectors, axis=0)

        image_vectors = (image_vectors > image_mean).astype(int)

        vrel_img = [image_vectors[x] for x, y in feedback.items() if y == "+R"]
        virel_img = [image_vectors[x] for x, y in feedback.items() if y == "+I"]
        rel_img = [image_vectors[x] for x, y in feedback.items() if y == "R"]
        irel_img = [image_vectors[x] for x, y in feedback.items() if y == "I"]


        fkvrel = (np.sum(vrel_img, axis=0) + 0.5)/(len(vrel_img) +1)
        fkvirel = (np.sum(virel_img,axis=0)+0.5)/(len(virel_img) +1)

        fkrel = (np.sum(rel_img, axis=0) + 0.5)/(len(rel_img) +1)
        fkirel = (np.sum(irel_img,axis=0)+0.5)/(len(irel_img) +1)

        numer = (0.7*fkvrel + 0.3*fkrel)/(1-(0.7*fkvrel + 0.3*fkrel))
        denom = (0.7*fkvirel + 0.3*fkirel)/(1-(0.7*fkvirel + 0.3*fkirel))
        print(np.where(numer > 1), np.where(numer == 0 ), np.where(numer == 1))
        dday = np.log(numer/denom)
        final = query_vector + beta*dday
        print(final)
        return final

    def svm_feedback_system(self,feedback, image_vectors, query_vector ):
        """Runs SVM for multiples relevance that we are given and then
          re ranks them based on the svm score or distances from the hyperplane"""
        
        # Data for all the images
        data = image_vectors
        # dim = dimension_reduction
        # data, _ = dim.nmf_als(data, 256)

        # temp = [4108, 2068, 4116, 6172, 6176, 2084, 4138, 2092, 2094, 4146, 6196, 8246, 6202, 6206, 6208, 4162, 4168, 6220, 6230, 4184, 4090, 2160, 116, 118, 6264, 124, 8318, 2178, 8324, 6278, 6282, 6286, 8338, 6292, 8340, 150, 154, 4258, 8362, 8364, 8368, 8370, 184, 6328, 8390, 4296, 4306, 216, 2264, 2270, 6368, 8418, 2280, 6378, 6386, 8190, 4344, 6398, 8452, 2310, 284, 286, 2336, 2340, 4388, 4392, 2346, 4398, 2358, 2366, 4416, 2374, 328, 4428, 6476, 6480, 342, 2392, 2398, 354, 2408, 366, 6510, 392, 2442, 6538, 6540, 8586, 8590, 6558, 4512, 2470, 2478, 2482, 436, 6580, 438, 6582, 6584, 6588, 2494, 4546, 8648, 460, 472, 8664, 8668, 6622, 492, 4590, 2546, 512, 2560, 2578, 4638, 6688, 4660, 2614, 6716, 6720, 578, 6732, 4698, 2654, 4706, 6754, 4710, 4722, 632, 2686, 2688, 6786, 644, 2694, 4748, 4754, 6802, 4756, 670, 2718, 4766, 680, 4782, 2740, 698, 6846, 706, 2760, 2764, 4820, 738, 4838, 6886, 744, 4842, 784, 6938, 6950, 822, 2870, 4918, 832, 4930, 836, 838, 842, 844, 4940, 6994, 4960, 2916, 884, 892, 2942, 4992, 900, 7052, 2962, 918, 7062, 7076, 934, 2982, 2992, 5040, 2994, 7090, 3002, 958, 962, 7114, 974, 7128, 988, 990, 996, 1002, 1004, 3054, 3056, 1016, 7174, 5128, 7180, 7184, 1052, 7210, 3116, 1072, 3122, 1078, 3126, 3128, 1082, 1090, 5196, 1102, 5200, 3154, 1112, 3166, 7262, 1122, 1124, 1126, 1128, 1146, 5252, 5254, 1172, 5274, 5284, 1190, 7338, 1200, 3258, 7354, 7364, 1222, 1226, 5328, 1240, 5338, 1274, 1276, 5372, 1282, 1286, 5400, 1316, 3382, 5432, 5436, 5444, 5448, 1358, 7508, 1366, 5462, 7514, 7522, 5478, 5496, 3450, 5502, 7556, 1414, 7562, 5516, 5540, 1446, 1450, 3510, 1466, 1470, 5578, 7636, 1496, 3554, 5608, 1514, 3570, 7668, 7682, 1540, 5640, 1546, 3594, 5648, 7698, 3604, 5660, 1574, 3624, 7720, 5680, 1596, 1620, 3702, 7806, 3718, 3728, 3744, 5796, 7846, 7848, 7852, 1712, 5808, 1724, 3772, 1726, 7868, 3776, 3780, 1738, 3788, 7884, 5838, 1746, 1762, 1770, 3820, 7934, 1792, 5892, 3850, 1806, 5908, 1818, 7964, 5924, 3892, 5954, 1864, 3914, 3922, 8018, 3924, 5972, 8020, 8022, 3934, 1892, 8038, 5992, 8050, 1910, 8056, 1916, 1942, 8092, 4002, 6050, 8106, 8120, 6074, 8122, 4032, 6086, 8142, 6104, 6106, 4068, 6138, 2046]
        # temp = [538, 1144, 1178, 1204, 1276, 1370, 1394, 1726, 1768, 1888, 1928, 1932, 1934, 1936, 1938, 1940, 1946, 1948, 1950, 1966, 1976, 1978, 1982, 1984, 1988, 1992, 1998, 2000, 2004, 2012, 2016, 2018, 2024, 2028, 2032, 2036, 2044, 2046, 2048, 2050, 2052, 2056, 2058, 2064, 2066, 2068, 2072, 2076, 2082, 2086, 2088, 2090, 2092, 2094, 2096, 2098, 2100, 2102, 2104, 2106, 2108, 2110, 2112, 2114, 2116, 2118, 2120, 2122, 2124, 2126, 2130, 2134, 2138, 2140, 2146, 2150, 2152, 2158, 2160, 2162, 2164, 2170, 2178, 2188, 2190, 2192, 2194, 2196, 2198, 2200, 2202, 2204, 2206, 2208, 2210, 2212, 2214, 2216, 2218, 2220, 2222, 2224, 2226, 2230, 2232, 2234, 2236, 2238, 2240, 2242, 2244, 2246, 2248, 2250, 2252, 2254, 2256, 2258, 2260, 2262, 2264, 2266, 2268, 2270, 2272, 2274, 2278, 2280, 2282, 2284, 2286, 2288, 2290, 2292, 2294, 2296, 2298, 2300, 2302, 2304, 2306, 2308, 2310, 2312, 2314, 2316, 2318, 2320, 2322, 2324, 2326, 2328, 2330, 2334, 2336, 2338, 2340, 2342, 2344, 2346, 2358, 2360, 2362, 2372, 2378, 2380, 2382, 2386, 2390, 2398, 2404, 2406, 2408, 2410, 2412, 2416, 2418, 2420, 2422, 2424, 2428, 2430, 2434, 2436, 2438, 2440, 2442, 2444, 2446, 2450, 2452, 2454, 2456, 2458, 2460, 2462, 2464, 2466, 2468, 2472, 2474, 2478, 2482, 2484, 2486, 2488, 2490, 2492, 2494, 2496, 2498, 2500, 2501, 2502, 2504, 2508, 2510, 2512, 2514, 2516, 2518, 2522, 2524, 2526, 2528, 2530, 2532, 2534, 2536, 2540, 2542, 2544, 2546, 2548, 2550, 2552, 2554, 2556, 2558, 2560, 2562, 2564, 2566, 2568, 2570, 2574, 2578, 2580, 2582, 2584, 2586, 2588, 2590, 2592, 2594, 2596, 2598, 2600, 2602, 2604, 2606, 2608, 2610, 2612, 2614, 2618, 2620, 2624, 2626, 2628, 2630, 2632, 2634, 2636, 2640, 2642, 2644, 2646, 2648, 2650, 2652, 2654, 2656, 2658, 2660, 2662, 2666, 2670, 2672, 2674, 2676, 2678, 2682, 2684, 2686, 2688, 2692, 2696, 2698, 2700, 2702, 2706, 2708, 2710, 2712, 2714, 2716, 2718, 2720, 2722, 2852, 2938, 2944, 2968, 3142, 3178, 3452, 3768, 4086, 4102, 4124, 4216, 4310, 4378, 4470, 4526, 4620, 4710, 4722, 4772, 5270, 5286, 5290, 5322, 5460, 5526, 5530, 5554, 5566, 5570, 5584, 5602, 5618, 6004, 6026, 6034, 6048, 6338, 6380, 6408, 6604, 6618, 6630, 6644, 6646, 7086, 7088, 7104, 7344, 7554, 7760, 7776, 7860, 7902, 7986, 8038, 8058, 8102, 8206, 8248, 8276, 8364, 8540, 8558]
        # task_4b = [x//2 for x in temp]
        task4b_considerset = self.task_4b["neighbour_images"]
        task4b_considerset = [x//2 for x in task4b_considerset]

        # Using 4 SVMs
        vrel_svm = svm_feedback.SVM()
        vr_i_index = vrel_svm.run_svm(condition_1= "'+R' in key", condition_2= "'+R' not in key", data=data, feedback=feedback, task4b_index=task4b_considerset)
        # Get weights
        vrel_w , b  = vrel_svm.return_weights_bias()
        
        rel_svm = svm_feedback.SVM()
        r_i_index = rel_svm.run_svm(condition_1= "'R' == key", condition_2= "'+R' != key", data=data, feedback=feedback, task4b_index=task4b_considerset)
        rel_w , b  = rel_svm.return_weights_bias()

        virel_svm = svm_feedback.SVM()
        vir_i_index = virel_svm.run_svm(condition_1= "'+I' in key", condition_2= "'+I' not in key", data=data, feedback=feedback, task4b_index=task4b_considerset)
        virel_w , b  = virel_svm.return_weights_bias()

        irel_svm = svm_feedback.SVM()
        ir_i_index = irel_svm.run_svm(condition_1= "'I' == key", condition_2= "'I' != key", data=data, feedback=feedback, task4b_index=task4b_considerset)
        irel_w, b = irel_svm.return_weights_bias()

        vrel, rel, vire, ire = [], [], [],[]
        for vr, r, vir, ir in zip(vr_i_index,r_i_index,vir_i_index,ir_i_index):
            max_variable = max([vr, r, vir, ir], key=lambda x: x[1])
            if max_variable == vr:

                vrel.append((max_variable[0], max_variable[1], "Very Relvent" ))
            elif max_variable == r:
                rel.append((max_variable[0], max_variable[1], "Relvent" ))
            elif max_variable == vir:
                vire.append((max_variable[0], max_variable[1], "Very IRelvent" ))
            elif max_variable == ir:
                ire.append((max_variable[0], max_variable[1], "IRelvent" )) 
        print(len(vrel), len(rel), len(vire), len(ire))
        vrel = sorted(vrel, key=lambda x: x[1], reverse=True)
        rel = sorted(rel, key=lambda x: x[1], reverse=True)
        vire = sorted(vire, key=lambda x: x[1], reverse=True)
        ire = sorted(ire, key=lambda x: x[1],reverse=True)

        final_output = vrel + rel + ire + vire
        final_output = [(x * 2, y, z) for x, y, z in final_output]
        i = 1
        for val in final_output:
            print("Rank - " + str(i) + " Image ID -  " + str(val[0]) + "  SVM Score - " + str(val[1]) + " Bucket - " + val[2])
            i += 1
            if i > 500:
                break
        # To see the images classified
        # output_list = vrel[:10].copy()
        # output_list = [(x * 2, y, z) for x, y, z in output_list]
        # utils.display_k_images_subplots(self.dataset,output_list, f"Using SVM for top 10 images Very Relevent")
        # output_list = vrel[-10: ].copy()
        # output_list = [(x * 2, y, z) for x, y, z in output_list]
        # utils.display_k_images_subplots(self.dataset,output_list, f"Using SVM for last 10 images  Bottom 10 in Very Relevent")

        # Adding the weights and remove the irelevant weights
        new_query_vector = query_vector + (0.7*vrel_w + 0.3*rel_w) - (0.7*virel_w + 0.3*irel_w)

        return new_query_vector

    def run_feedback(self, query_vector = []):
        print("=" * 25, "MENU", "=" * 25)

        # Assumption: The model runs with fc_layer
        option = 5

        image_vectors = mongo_query.get_all_feature_descriptor(
            utils.feature_model[option]
        )

        # get user feedback 
        feedback = self.get_user_feedback()
        option = utils.get_user_selection_relevance_feedback()
        # 1 -> SVM
        # 2 -> Probabilist Relevance Feedback System
        # Testing
        # feedback = { 2494//2 : "+R", 2482//2: "+R", 2084//2 : "R", 2092//2 : "R", 4138//2 : "I", 4116//2 : "I", 6202//2 : "+I" , 6206//2 : "+I"}
        # feedback= {2500//2: "+R",  2308//2 : "+R",  2252//2 : "R", 5430//2 : "I", 7538//2 : "+I", 8140//2 : "+I",  8162//2 : "I"}
        res = None
        if len(query_vector) == 0 :
            query_vector = image_vectors[self.task_4b["query_image"]//2]
        match option:
            case 1:
                
                res = self.svm_feedback_system(
                   feedback, image_vectors, query_vector
                   )
            case 2:
                
                res = self.probaility_feedback(
                   feedback, image_vectors, query_vector
                   )
        return res
        # t3.print_labels(res)


if __name__ == "__main__":
    
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
        task4 = Task4.Task4b()
        query_vector = task4.runTask4b(imageID = task_4b["query_image"], query_vector = new_vector)
        exit_condition = input("To exit press 0 else 1 -  ")
        if exit_condition.lower() == '0':
            break

