FILE FRAME OF THE SYSTEM
+sound_detection                    #This project is aimed to learn and classify whether the test sound are the yell of epilepsy patients or not.
    +old_data                       #This folder contains the old original sound and its data after some process.
        +positive                   #This folder contains the positive samples which only contains the yell of patients
            positive.wav            #This wave file is cut and merged before manually, and its duration is about 70 seconds.
            +500_100                #This folder contains the samples cut by sampling 500 and overlapping 100 method, and the images of each of them as well.
            +750_200                #This folder contains the samples cut by sampling 750 and overlapping 200 method, and the images of each of them as well.
            +1000_500               #This folder contains the samples cut by sampling 1000 and overlapping 500 method, and the images of each of them as well.
        +negative                   #The same as positive folder except the sound to be deal with.
            negative.wav            #This file contains the several kinds of noises of old data.
            +500_100
            +750_200
            +1000_500
        +old_data_pkl               #This folder contains the training data and test data.
            train_500_100.pkl
            train_750_200.pkl
            train_1000_500.pkl
            test_500_100.pkl
            test_750_200.pkl
            test_1000_500.pkl
    +new_data                       #This folder contains the new original sound and its data after some process.
        +positive
            positive.wav            #This file only contains the yell of patients of the new data.
            positive2.wav           #This file contains the yell of patients of the new data and combine some of the old data.
            +500_100
            +750_200
            +1000_500
            +500_100_2              #This and 2 below folders is corresponding to the positive2.wav
            +750_200_2
            +1000_500_2
        +negative
            negative.wav            #This file contains several kinds of noises of new data.
            negative2.wav           #This file contains several kinds of noises combining old data and new data.
            +500_100
            +750_200
            +1000_500
            +500_100_2              #This and 2 below folders is corresponding to the negative2.wav
            +750_200_2
            +1000_500_2
        +new_data_pkl
            train_500_100.pkl
            train_750_200.pkl
            train_1000_500.pkl
            test_500_100.pkl
            test_750_200.pkl
            test_1000_500.pkl
        +new_data2_pkl
            train_500_100_2.pkl
            train_750_200_2.pkl
            train_1000_500_2.pkl
            test_500_100_2.pkl
            test_750_200_2.pkl
            test_1000_500_2.pkl
        cut_by_samples.py          # This python script cuts the sound which cut and merged before manually by sampling and save them as short samples.
        to_img.py                  # This python script converts wave files to image files.
        check_empty_file.py        # This python script checks the image file whether they are empty, and save the index of empty files.
        to_img_single.py           # This python script converts empty files to image files.
        img_features.py            # This python script extracts image features and randomly separated into training data and test data.
        inputimg.py                # This python script inputs image features into CNN model to train and classify the test data.
